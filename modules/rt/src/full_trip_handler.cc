#include "motis/rt/full_trip_handler.h"

#include <cassert>
#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "utl/enumerate.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/access/station_access.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/rt/build_route_node.h"
#include "motis/rt/connection_builder.h"
#include "motis/rt/incoming_edges.h"
#include "motis/rt/update_constant_graph.h"

using namespace motis::logging;
using namespace motis::ris;

namespace motis::rt {

inline std::string_view view(flatbuffers::String const* s) {
  return {s->c_str(), s->size()};
}

struct full_trip_handler {
  struct section {
    light_connection lc_;
    station_node* from_{};
    station_node* to_{};
    bool in_allowed_{};
    bool out_allowed_{};
  };

  full_trip_handler(statistics& stats, schedule& sched,
                    update_msg_builder& update_builder,
                    FullTripMessage const* msg)
      : stats_{stats},
        sched_{sched},
        update_builder_{update_builder},
        msg_{msg} {}

  void handle_msg() {
    if (!check_events()) {
      return;
    }

    get_or_add_station(msg_->trip_id()->start_station());
    get_or_add_station(msg_->trip_id()->target_station());
    auto const ftid = get_full_trip_id();
    auto const existing_trp = find_existing_trip(ftid);
    if (existing_trp != nullptr) {
      return;  // NYI
    }

    auto const sections = build_sections();
    auto incoming = std::vector<incoming_edge_patch>{};
    save_outgoing_edges(get_station_nodes(sections), incoming);
    auto const route = build_route(sections, incoming);
    patch_incoming_edges(incoming);
    result_.trp_ = create_trip(ftid, route);

    for (auto const station_idx : result_.stations_addded_) {
      update_builder_.add_station(station_idx);
    }
    update_builder_.add_reroute(result_.trp_, {}, 0);
  }

private:
  station_node* get_or_add_station(StationInfo const* si) {
    if (auto const sn = find_station_node(sched_, view(si->eva()));
        sn != nullptr) {
      return sn;
    } else {
      return add_station(si);
    }
  }

  station_node* add_station(StationInfo const* si) {
    auto const index = static_cast<node_id_t>(sched_.stations_.size());
    utl::verify(index == sched_.station_nodes_.size(),
                "stations != station_nodes");
    utl::verify(index < sched_.non_station_node_offset_,
                "station limit reached");

    auto const st =
        sched_.stations_.emplace_back(mcd::make_unique<station>()).get();
    st->index_ = index;
    st->name_ = si->name()->str();
    st->eva_nr_ = si->eva()->str();
    st->transfer_time_ = 2;
    st->equivalent_.push_back(st);

    sched_.eva_to_station_.emplace(st->eva_nr_, st);
    if (si->ds100()->size() != 0) {
      sched_.ds100_to_station_.emplace(si->ds100()->str(), st);
    }

    auto& sn = sched_.station_nodes_.emplace_back(
        mcd::make_unique<station_node>(make_station_node(index)));
    constant_graph_add_station_node(sched_);

    result_.stations_addded_.emplace_back(index);

    return sn.get();
  }

  full_trip_id get_full_trip_id() {
    full_trip_id ftid;

    auto const tid = msg_->trip_id()->id();
    ftid.primary_.station_id_ =
        get_station(sched_, view(tid->station_id()))->index_;
    ftid.primary_.time_ = unix_to_motistime(sched_, tid->time());
    ftid.primary_.train_nr_ = tid->train_nr();
    ftid.secondary_.target_station_id_ =
        get_station(sched_, view(tid->target_station_id()))->index_;
    ftid.secondary_.target_time_ = unix_to_motistime(sched_, tid->time());
    ftid.secondary_.line_id_ = view(tid->line_id());

    return ftid;
  }

  trip* find_existing_trip(full_trip_id const& ftid) {
    auto it = std::lower_bound(
        begin(sched_.trips_), end(sched_.trips_),
        std::make_pair(ftid.primary_, static_cast<trip*>(nullptr)));
    if (it == end(sched_.trips_) || !(it->first == ftid.primary_)) {
      return nullptr;
    }
    for (; it != end(sched_.trips_) && it->first == ftid.primary_; ++it) {
      if (it->second->id_.secondary_ == ftid.secondary_) {
        return it->second;
      }
    }
    return nullptr;
  }

  bool check_events() {
    auto last_arrival_station = std::string_view{};
    auto last_arrival_time = static_cast<time>(0);
    for (auto const& sec : *msg_->sections()) {
      auto const departure_station = view(sec->departure()->station()->eva());
      auto const arrival_station = view(sec->arrival()->station()->eva());
      if (!last_arrival_station.empty() &&
          last_arrival_station != departure_station) {
        result_.status_ = full_trip_result::status::INVALID_STATION_SEQUENCE;
        return false;
      }
      auto const departure_time =
          unix_to_motistime(sched_, sec->departure()->schedule_time());
      auto const arrival_time =
          unix_to_motistime(sched_, sec->arrival()->schedule_time());
      if (departure_time > arrival_time || departure_time < last_arrival_time) {
        result_.status_ =
            full_trip_result::status::INVALID_SCHEDULE_TIME_SEQUENCE;
        return false;
      }
      last_arrival_station = arrival_station;
      last_arrival_time = arrival_time;
    }
    return true;
  }

  std::vector<section> build_sections() {
    return utl::to_vec(*msg_->sections(), [&](TripSection const* ts) {
      auto const from = get_or_add_station(ts->departure()->station());
      auto const to = get_or_add_station(ts->arrival()->station());
      auto lc = light_connection{};
      lc.valid_ = 1U;
      // TODO(pablo): current_time
      lc.d_time_ = unix_to_motistime(sched_, ts->departure()->schedule_time());
      lc.a_time_ = unix_to_motistime(sched_, ts->arrival()->schedule_time());
      lc.full_con_ = get_full_con(
          sched_, con_infos_, ts->departure()->current_track()->str(),
          ts->arrival()->current_track()->str(), ts->category()->code()->str(),
          ts->line_id()->str(), ts->train_nr());
      return section{lc, from, to, ts->departure()->interchange_allowed(),
                     ts->arrival()->interchange_allowed()};
    });
  }

  static std::set<station_node*> get_station_nodes(
      std::vector<section> const& sections) {
    auto nodes = std::set<station_node*>{};
    for (auto const& sec : sections) {
      nodes.insert(sec.from_);
      nodes.insert(sec.to_);
    }
    return nodes;
  }

  std::vector<node*> build_route_nodes(
      std::vector<section> const& sections, std::uint32_t route_id,
      std::vector<incoming_edge_patch>& incoming) {
    assert(!sections.empty());
    auto nodes = std::vector<node*>{};
    nodes.reserve(sections.size() + 1);

    auto const build_node = [&](station_node* sn, bool in_allowed,
                                bool out_allowed) {
      return build_route_node(sched_, route_id, sched_.next_node_id_++, sn,
                              sched_.stations_.at(sn->id_)->transfer_time_,
                              in_allowed, out_allowed, incoming);
    };

    auto const& first_section = sections.front();
    nodes.emplace_back(
        build_node(first_section.from_, first_section.in_allowed_, true));
    for (auto i = 1ULL; i < sections.size(); ++i) {
      nodes.emplace_back(build_node(sections[i].from_, sections[i].in_allowed_,
                                    sections[i - 1].out_allowed_));
    }
    auto const& last_section = sections.back();
    nodes.emplace_back(
        build_node(last_section.to_, true, last_section.out_allowed_));

    assert(nodes.size() == sections.size() + 1);
    return nodes;
  }

  mcd::vector<trip::route_edge> build_route(
      std::vector<section> const& sections,
      std::vector<incoming_edge_patch>& incoming) {
    if (sections.empty()) {
      return {};
    }
    auto trip_edges = mcd::vector<trip::route_edge>{};
    auto const route_id = sched_.route_count_++;

    auto const route_nodes = build_route_nodes(sections, route_id, incoming);

    for (auto const& [i, sec] : utl::enumerate(sections)) {
      auto const from_route_node = route_nodes[i];  // NOLINT
      auto const to_route_node = route_nodes[i + 1];  // NOLINT

      auto const& route_edge = from_route_node->edges_.emplace_back(
          make_route_edge(from_route_node, to_route_node, {sec.lc_}));
      add_outgoing_edge(&route_edge, incoming);
      trip_edges.emplace_back(&route_edge);
      constant_graph_add_route_edge(sched_, &route_edge);
    }

    return trip_edges;
  }

  trip* create_trip(full_trip_id const& ftid,
                    mcd::vector<trip::route_edge> const& trip_edges) {
    auto const edges =
        sched_.trip_edges_
            .emplace_back(
                mcd::make_unique<mcd::vector<trip::route_edge>>(trip_edges))
            .get();

    auto const lcon_idx = static_cast<lcon_idx_t>(0U);
    auto const trp = sched_.trip_mem_
                         .emplace_back(mcd::make_unique<trip>(
                             ftid, edges, lcon_idx, trip_debug{}))
                         .get();

    auto const trp_entry = mcd::pair{ftid.primary_, ptr<trip>(trp)};
    sched_.trips_.insert(
        std::lower_bound(begin(sched_.trips_), end(sched_.trips_), trp_entry),
        trp_entry);

    auto const new_trps_id = sched_.merged_trips_.size();
    sched_.merged_trips_.emplace_back(
        mcd::make_unique<mcd::vector<ptr<trip>>,
                         std::initializer_list<ptr<trip>>>({trp}));

    for (auto const& trp_edge : trip_edges) {
      trp_edge.get_edge()->m_.route_edge_.conns_[lcon_idx].trips_ = new_trps_id;
    }

    return trp;
  }

public:
  full_trip_result get_result() { return result_; }

  statistics& stats_;
  schedule& sched_;
  update_msg_builder& update_builder_;
  ris::FullTripMessage const* msg_;
  full_trip_result result_;
  std::map<connection_info, connection_info const*> con_infos_;
};

full_trip_result handle_full_trip_msg(statistics& stats, schedule& sched,
                                      update_msg_builder& update_builder,
                                      ris::FullTripMessage const* msg) {
  auto handler = full_trip_handler{stats, sched, update_builder, msg};
  handler.handle_msg();
  return handler.get_result();
}

}  // namespace motis::rt
