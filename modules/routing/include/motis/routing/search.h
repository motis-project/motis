#pragma once

#include "utl/to_vec.h"

#include "motis/hash_map.h"

#include "motis/core/common/timing.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/validate_graph.h"
#include "motis/core/journey/print_trip.h"
#include "motis/routing/lower_bounds.h"
#include "motis/routing/output/labels_to_journey.h"
#include "motis/routing/pareto_dijkstra.h"

namespace motis::routing {

struct search_query {
  schedule const* sched_{nullptr};
  mem_manager* mem_{nullptr};
  node const* from_{nullptr};
  station_node const* to_{nullptr};
  time interval_begin_{0};
  time interval_end_{0};
  bool extend_interval_earlier_{false};
  bool extend_interval_later_{false};
  std::vector<edge> query_edges_;
  unsigned min_journey_count_{0};
  bool use_start_metas_{false};
  bool use_dest_metas_{false};
  bool use_start_footpaths_{false};
  light_connection const* lcon_{nullptr};
};

struct search_result {
  search_result() = default;
  search_result(statistics stats, std::vector<journey> journeys,
                time interval_begin, time interval_end)
      : stats_(std::move(stats)),  // NOLINT
        journeys_(std::move(journeys)),
        interval_begin_(interval_begin),
        interval_end_(interval_end) {}
  explicit search_result(unsigned travel_time_lb) : stats_(travel_time_lb) {}
  statistics stats_;
  std::vector<journey> journeys_;
  time interval_begin_{INVALID_TIME};
  time interval_end_{INVALID_TIME};
};

duration get_fastest_direct_with_foot(schedule const& sched,
                                      search_query const& q,
                                      search_dir const dir) {
  auto min = std::numeric_limits<duration>::max();
  for (auto const& start : q.query_edges_) {
    auto const& start_station =
        *sched.stations_.at(start.get_destination(dir)->id_);
    auto const& footpaths =
        (dir == search_dir::FWD ? start_station.outgoing_footpaths_
                                : start_station.incoming_footpaths_);
    for (auto const& fp : footpaths) {
      auto const fp_target =
          dir == search_dir::FWD ? fp.to_station_ : fp.from_station_;
      for (auto const& dest : q.query_edges_) {
        if (dest.get_source(dir)->id_ == fp_target) {
          min = std::min(
              min, static_cast<duration>(start.get_foot_edge_cost().time_ +
                                         fp.duration_ +
                                         dest.get_foot_edge_cost().time_));
        }
      }
    }
  }
  return min;
}

duration get_fastest_start_dest_overlap(search_query const& q,
                                        search_dir const dir) {
  auto min = std::numeric_limits<duration>::max();
  for (auto const& start : q.query_edges_) {
    auto const start_station = start.get_destination(dir);
    for (auto const& dest : q.query_edges_) {
      auto const dest_station = dest.get_source(dir);
      if (start_station == dest_station) {
        min = std::min(min,
                       static_cast<duration>(start.get_foot_edge_cost().time_ +
                                             dest.get_foot_edge_cost().time_));
      }
    }
  }
  return min;
}

duration get_fastest_direct(schedule const& sched, search_query const& q,
                            search_dir const dir) {
  return std::min(get_fastest_direct_with_foot(sched, q, dir),
                  get_fastest_start_dest_overlap(q, dir));
}

template <search_dir Dir, typename StartLabelGenerator, typename Label>
struct search {
  static search_result get_connections(search_query const& q) {
    mcd::hash_map<unsigned, std::vector<simple_edge>>
        travel_time_lb_graph_edges;
    mcd::hash_map<unsigned, std::vector<simple_edge>> transfers_lb_graph_edges;
    auto const route_offset = q.sched_->non_station_node_offset_;
    for (auto const& e : q.query_edges_) {
      auto const from_node = (Dir == search_dir::FWD) ? e.from_ : e.to_;
      auto const to_node = (Dir == search_dir::FWD) ? e.to_ : e.from_;

      // station graph
      auto const from_station = from_node->get_station()->id_;
      auto const to_station = to_node->get_station()->id_;

      // interchange graph
      auto const from_interchange = from_node->is_route_node()
                                        ? route_offset + from_node->route_
                                        : from_station;
      auto const to_interchange = to_node->is_route_node()
                                      ? route_offset + to_node->route_
                                      : to_station;

      auto const ec = e.get_minimum_cost();

      travel_time_lb_graph_edges[to_station].emplace_back(
          simple_edge{from_station, ec.time_});
      transfers_lb_graph_edges[to_interchange].emplace_back(simple_edge{
          from_interchange, static_cast<uint16_t>(ec.transfer_ ? 1 : 0)});
    }

    auto const& meta_goals = q.sched_->stations_[q.to_->id_]->equivalent_;
    std::vector<int> goal_ids;
    std::vector<bool> is_goal(q.sched_->stations_.size(), false);
    for (auto const& meta_goal : meta_goals) {
      goal_ids.push_back(meta_goal->index_);
      is_goal[meta_goal->index_] = true;
      if (!q.use_dest_metas_) {
        break;
      }
    }
    if (q.to_ == q.sched_->station_nodes_.at(1).get()) {
      goal_ids.push_back(q.to_->id_);
      is_goal[q.to_->id_] = true;
    }

    lower_bounds lbs(
        *q.sched_,  //
        Dir == search_dir::FWD ? q.sched_->travel_time_lower_bounds_fwd_
                               : q.sched_->travel_time_lower_bounds_bwd_,
        Dir == search_dir::FWD ? q.sched_->transfers_lower_bounds_fwd_
                               : q.sched_->transfers_lower_bounds_bwd_,
        goal_ids, travel_time_lb_graph_edges, transfers_lb_graph_edges);

    MOTIS_START_TIMING(travel_time_lb_timing);
    lbs.travel_time_.run();
    MOTIS_STOP_TIMING(travel_time_lb_timing);

    auto const reachable_start = [&]() {
      if (lbs.travel_time_.is_reachable(lbs.travel_time_[q.from_])) {
        return true;
      } else if (!q.use_start_metas_ ||
                 q.from_->id_ >= q.sched_->stations_.size()) {
        return false;
      } else {
        auto const& meta_froms = q.sched_->stations_[q.from_->id_]->equivalent_;
        return std::any_of(
            begin(meta_froms), end(meta_froms), [&](station const* meta) {
              auto const* sn = q.sched_->station_nodes_.at(meta->index_).get();
              return lbs.travel_time_.is_reachable(lbs.travel_time_[sn]);
            });
      }
    };

    if (!reachable_start()) {
      return search_result(MOTIS_TIMING_MS(travel_time_lb_timing));
    }

    MOTIS_START_TIMING(transfers_lb_timing);
    lbs.transfers_.run();
    MOTIS_STOP_TIMING(transfers_lb_timing);

    auto const create_start_edge = [&](node* to) {
      return Dir == search_dir::FWD ? make_foot_edge(nullptr, to)
                                    : make_foot_edge(to, nullptr);
    };
    auto mutable_node = const_cast<node*>(q.from_);  // NOLINT
    auto const start_edge = create_start_edge(mutable_node);

    std::vector<edge> meta_edges;
    if (q.from_->is_route_node() ||
        q.from_ == q.sched_->station_nodes_.at(0).get()) {
      if (!lbs.travel_time_.is_reachable(lbs.travel_time_[q.from_])) {
        return search_result(MOTIS_TIMING_MS(travel_time_lb_timing));
      }
    } else if (!q.use_start_metas_) {
      if (!lbs.travel_time_.is_reachable(lbs.travel_time_[q.from_])) {
        return search_result(MOTIS_TIMING_MS(travel_time_lb_timing));
      }
      meta_edges.push_back(start_edge);
    } else {
      utl::verify(q.from_->id_ < q.sched_->stations_.size(),
                  "invalid start node with meta station search");
      auto const& meta_froms = q.sched_->stations_[q.from_->id_]->equivalent_;
      for (auto const& meta_from : meta_froms) {
        auto meta_edge = create_start_edge(
            q.sched_->station_nodes_[meta_from->index_].get());
        meta_edges.push_back(meta_edge);
      }
    }

    mcd::hash_map<node const*, std::vector<edge>> additional_edges;
    for (auto const& e : q.query_edges_) {
      additional_edges[e.get_source<Dir>()].push_back(e);
    }

    auto const fastest_direct = get_fastest_direct(*q.sched_, q, Dir);
    pareto_dijkstra<Dir, Label, lower_bounds> pd(
        *q.sched_, q.sched_->next_node_id_, q.sched_->stations_.size(), is_goal,
        std::move(additional_edges), fastest_direct, lbs, *q.mem_);

    auto const add_start_labels = [&](time interval_begin, time interval_end) {
      pd.add_start_labels(StartLabelGenerator::generate(
          *q.sched_, *q.mem_, lbs, &start_edge, meta_edges, q.query_edges_,
          interval_begin, interval_end, q.lcon_, q.use_start_footpaths_,
          fastest_direct));
    };

    time const schedule_begin = SCHEDULE_OFFSET_MINUTES;
    time const schedule_end =
        (q.sched_->schedule_end_ - q.sched_->schedule_begin_) / 60;

    auto const map_to_interval = [&schedule_begin, &schedule_end](time t) {
      return std::min(schedule_end, std::max(schedule_begin, t));
    };

    add_start_labels(q.interval_begin_, q.interval_end_);

    MOTIS_START_TIMING(pareto_dijkstra_timing);
    auto max_interval_reached = false;
    auto interval_begin = q.interval_begin_;
    auto interval_end = q.interval_end_;

    auto const departs_in_interval = [](Label const* l,
                                        motis::time interval_begin,
                                        motis::time interval_end) {
      return interval_end == INVALID_TIME ||  // ontrip
             (l->start_ >= interval_begin && l->start_ <= interval_end);
    };

    auto const number_of_results_in_interval =
        [&interval_begin, &interval_end,
         departs_in_interval](std::vector<Label*> const& labels) {
          return std::count_if(begin(labels), end(labels), [&](Label const* l) {
            return departs_in_interval(l, interval_begin, interval_end);
          });
        };

    auto search_iterations = 0UL;

    while (!max_interval_reached) {
      max_interval_reached =
          (!q.extend_interval_earlier_ || interval_begin == schedule_begin) &&
          (!q.extend_interval_later_ || interval_end == schedule_end);

      pd.search();
      ++search_iterations;

      if (number_of_results_in_interval(pd.get_results()) >=
          q.min_journey_count_) {
        break;
      }

      auto const new_interval_begin = q.extend_interval_earlier_
                                          ? map_to_interval(interval_begin - 60)
                                          : interval_begin;
      auto const new_interval_end = q.extend_interval_later_
                                        ? map_to_interval(interval_end + 60)
                                        : interval_end;

      if (interval_begin != schedule_begin) {
        add_start_labels(new_interval_begin,
                         map_to_interval(interval_begin - 1));
      }

      if (interval_end != schedule_end) {
        add_start_labels(map_to_interval(interval_end + 1), new_interval_end);
      }

      interval_begin = new_interval_begin;
      interval_end = new_interval_end;
    }
    MOTIS_STOP_TIMING(pareto_dijkstra_timing);

    auto stats = pd.get_statistics();
    stats.travel_time_lb_ = MOTIS_TIMING_MS(travel_time_lb_timing);
    stats.transfers_lb_ = MOTIS_TIMING_MS(transfers_lb_timing);
    stats.pareto_dijkstra_ = MOTIS_TIMING_MS(pareto_dijkstra_timing);
    stats.interval_extensions_ = search_iterations - 1;

    auto filtered = pd.get_results();
    filtered.erase(std::remove_if(begin(filtered), end(filtered),
                                  [&](Label const* l) {
                                    return !departs_in_interval(
                                        l, interval_begin, interval_end);
                                  }),
                   end(filtered));

    return search_result(stats,
                         utl::to_vec(filtered,
                                     [&q](Label* label) {
                                       return output::labels_to_journey(
                                           *q.sched_, label, Dir);
                                     }),
                         interval_begin, interval_end);
  }
};

}  // namespace motis::routing
