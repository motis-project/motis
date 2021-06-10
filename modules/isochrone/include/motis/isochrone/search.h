#pragma once

#include "utl/to_vec.h"

#include "motis/hash_map.h"

#include "motis/core/common/timing.h"
#include "motis/core/schedule/schedule.h"
#include "motis/isochrone/lower_bounds.h"
#include "motis/isochrone/output/labels_to_journey.h"
#include "motis/isochrone/pareto_dijkstra.h"

namespace motis::isochrone {

struct search_query {
  schedule const* sched_{nullptr};
  mem_manager* mem_{nullptr};
  node const* from_{nullptr};
  time interval_begin_{0};
  time interval_end_{0};
  bool use_start_footpaths_{false};
  light_connection const* lcon_{nullptr};
};

struct search_result {
  search_result() = default;
  search_result(statistics stats, std::vector<station> stations,
                std::vector<long> travel_times, time interval_begin,
                time interval_end)
      : stats_(std::move(stats)),  // NOLINT
        stations_(std::move(stations)),
        travel_times_(std::move(travel_times)),
        interval_begin_(interval_begin),
        interval_end_(interval_end) {}
  explicit search_result(unsigned travel_time_lb) : stats_(travel_time_lb) {}
  statistics stats_;
  std::vector<station> stations_;
  std::vector<long> travel_times_;
  time interval_begin_{INVALID_TIME};
  time interval_end_{INVALID_TIME};
};

template <search_dir Dir, typename StartLabelGenerator, typename Label>
struct search {
  static search_result get_connections(search_query const& q) {
    mcd::hash_map<unsigned, std::vector<simple_edge>>
        travel_time_lb_graph_edges;
    mcd::hash_map<unsigned, std::vector<simple_edge>> transfers_lb_graph_edges;
    auto const route_offset = q.sched_->station_nodes_.size();
    for (auto const& e : q.query_edges_) {
      auto const from_node = e.from_;
      auto const to_node = e.to_;

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

    auto const& meta_goals = q.sched_->stations_[q.from_->id_]->equivalent_;
    std::vector<int> goal_ids;
    boost::container::vector<bool> is_goal(q.sched_->stations_.size(), false);

    for (auto const& meta_goal : meta_goals) {
      goal_ids.push_back(meta_goal->index_);
      is_goal[meta_goal->index_] = true;
      if (!q.use_dest_metas_) {
        break;
      }
    }


    auto const create_start_edge = [&](node* to) {
      return make_foot_edge(nullptr, to)

    };
    auto mutable_node = const_cast<node*>(q.from_);  // NOLINT
    auto const start_edge = create_start_edge(mutable_node);

    std::vector<edge> meta_edges;

    mcd::hash_map<node const*, std::vector<edge>> additional_edges;
    for (auto const& e : q.query_edges_) {
      additional_edges[e.get_source<Dir>()].push_back(e);
    }

    et_dijkstra ed(
        q.sched_->next_node_id_, q.sched_->stations_.size(), is_goal,
        std::move(additional_edges), q.interval_end_, *q.mem_);


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

    std::vector<station> stations;
    std::vector<long> travel_times;

    for (auto label : filtered) {
      if (label->get_node()->is_route_node()) {
        stations.emplace_back(
            *q.sched_->stations_[label->edge_->to_->get_station()->id_]);
        travel_times.emplace_back(
            motis_to_unixtime(q.sched_->schedule_begin_, q.interval_end_) -
            motis_to_unixtime(q.sched_->schedule_begin_, label->now_));
      }
    }

    return search_result(stats, stations, travel_times,

                         interval_begin, interval_end);
  }
};

}  // namespace motis::isochrone
