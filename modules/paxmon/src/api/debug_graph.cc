#include "motis/paxmon/api/debug_graph.h"

#include <algorithm>
#include <set>
#include <vector>

#include "utl/get_or_create.h"
#include "utl/to_vec.h"

#include "motis/data.h"
#include "motis/hash_map.h"

#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace flatbuffers;

namespace motis::paxmon::api {

struct selected_graph_indices {
  explicit selected_graph_indices(universe const& uv) : uv_{uv} {}

  void add_node(event_node_index const ni, bool const with_edges) {
    node_indices_.insert(ni);
    if (with_edges) {
      add_node_edges(&uv_.graph_.nodes_.at(ni));
    }
  }

  void add_node(event_node const* n, bool const with_edges) {
    node_indices_.insert(n->index_);
    if (with_edges) {
      add_node_edges(n);
    }
  }

  void add_edge(edge_index const ei, bool with_adjacent_edges) {
    edge_indices_.insert(ei);
    add_edge_endpoints(ei.get(uv_), with_adjacent_edges);
  }

  void add_edge(edge const* e, bool const with_adjacent_edges) {
    edge_indices_.insert(get_edge_index(uv_, e));
    add_edge_endpoints(e, with_adjacent_edges);
  }

  void add_edge_endpoints(edge const* e, bool const with_adjacent_edges) {
    add_node(e->from(uv_), with_adjacent_edges);
    add_node(e->to(uv_), with_adjacent_edges);
  }

  void add_node_edges(event_node const* n) {
    for (auto const& e : n->outgoing_edges(uv_)) {
      add_edge(&e, false);
    }
    for (auto const& e : n->incoming_edges(uv_)) {
      add_edge(&e, false);
    }
  }

  universe const& uv_;
  std::set<event_node_index> node_indices_;
  std::set<edge_index> edge_indices_;
};

msg_ptr debug_graph(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonDebugGraphRequest, msg);
  auto const uv_access =
      get_universe_and_schedule(data, req->universe(), ctx::access_t::READ);
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;

  auto const req_pgwrs =
      utl::to_vec(*req->group_routes(), [](PaxMonGroupWithRouteId const* pgwr) {
        return passenger_group_with_route{pgwr->g(), pgwr->r()};
      });
  auto const req_filter_groups = req->filter_groups();
  auto const include_full_trips_from_group_routes =
      req->include_full_trips_from_group_routes();
  auto const include_canceled_trip_nodes = req->include_canceled_trip_nodes();

  auto selected = selected_graph_indices{uv};

  auto const add_trip = [&](trip_idx_t const trip_idx) {
    auto const tdi = uv.trip_data_.get_index(trip_idx);
    for (auto const& ei : uv.trip_data_.edges(tdi)) {
      selected.add_edge(ei, true);
    }
    if (include_canceled_trip_nodes) {
      for (auto const& ni : uv.trip_data_.canceled_nodes(tdi)) {
        selected.add_node(ni, true);
      }
    }
  };

  for (auto const node_idx : *req->node_indices()) {
    selected.add_node(node_idx, true);
  }

  for (auto const& pgwr : req_pgwrs) {
    auto const& route = uv.passenger_groups_.route(pgwr);
    for (auto const& ei :
         uv.passenger_groups_.route_edges(route.edges_index_)) {
      selected.add_edge(ei, true);
    }
    if (include_full_trips_from_group_routes) {
      for (auto const& leg :
           uv.passenger_groups_.journey(route.compact_journey_index_).legs()) {
        add_trip(leg.trip_idx_);
      }
    }
  }

  for (auto const& fbs_trp : *req->trip_ids()) {
    auto const* trp = from_fbs(sched, fbs_trp);
    add_trip(trp->trip_idx_);
  }

  message_creator mc;
  auto fbs_trips = std::vector<Offset<TripServiceInfo>>{};
  auto trips_map = mcd::hash_map<ptr<trip>, unsigned>{};

  auto fbs_nodes =
      utl::to_vec(selected.node_indices_, [&](auto const node_idx) {
        auto const& n = uv.graph_.nodes_.at(node_idx);
        auto const log =
            node_idx < uv.graph_log_.node_log_.index_size()
                ? utl::to_vec(uv.graph_log_.node_log_.at(node_idx),
                              [&](auto const& entry) {
                                return CreatePaxMonDebugNodeLogEntry(
                                    mc, to_fbs_time(sched, entry.system_time_),
                                    to_fbs_time(sched, entry.node_time_),
                                    entry.valid_);
                              })
                : std::vector<Offset<PaxMonDebugNodeLogEntry>>{};
        return CreatePaxMonDebugNode(
            mc, n.index_, to_fbs_time(sched, n.schedule_time_),
            to_fbs_time(sched, n.time_), n.type_ == event_type::ARR, n.valid_,
            to_fbs(mc, n.get_station(sched)), mc.CreateVector(log));
      });

  auto const get_fbs_group_routes = [&](pci_index const pci) {
    if (req_filter_groups) {
      auto routes = std::vector<PaxMonGroupRouteBaseInfo>{};
      for (auto const& pgwr : uv.pax_connection_info_.group_routes(pci)) {
        if (std::find(req_pgwrs.begin(), req_pgwrs.end(), pgwr) !=
            req_pgwrs.end()) {
          routes.emplace_back(to_fbs_base_info(mc, uv.passenger_groups_, pgwr));
        }
      }
      return routes;
    } else {
      return utl::to_vec(
          uv.pax_connection_info_.group_routes(pci), [&](auto const& pgwr) {
            return to_fbs_base_info(mc, uv.passenger_groups_, pgwr);
          });
    }
  };

  auto const get_fbs_pax_log = [&](pci_index const pci) {
    auto pax_log = std::vector<Offset<PaxMonDebugPaxLogEntry>>{};
    if (pci >= uv.graph_log_.pci_log_.index_size()) {
      return pax_log;
    }
    for (auto const& entry : uv.graph_log_.pci_log_.at(pci)) {
      if (!req_filter_groups || std::find(req_pgwrs.begin(), req_pgwrs.end(),
                                          entry.pgwr_) != req_pgwrs.end()) {
        auto const fbs_pgwr =
            PaxMonGroupWithRouteId{entry.pgwr_.pg_, entry.pgwr_.route_};
        pax_log.emplace_back(CreatePaxMonDebugPaxLogEntry(
            mc, to_fbs_time(sched, entry.system_time_),
            static_cast<PaxMonDebugPaxLogAction>(entry.action_),
            static_cast<PaxMonDebugPaxLogReason>(entry.reason_), &fbs_pgwr));
      }
    }
    return pax_log;
  };

  auto const get_trip_indices = [&](edge const* e) {
    if (!e->has_trips()) {
      return std::vector<unsigned>{};
    }
    return utl::to_vec(e->get_trips(sched), [&](auto const& trp) {
      return utl::get_or_create(trips_map, trp, [&]() {
        auto const idx = static_cast<unsigned>(fbs_trips.size());
        fbs_trips.emplace_back(to_fbs_trip_service_info(mc, sched, trp));
        return idx;
      });
    });
  };

  auto fbs_edges =
      utl::to_vec(selected.edge_indices_, [&](auto const& edge_idx) {
        auto const* e = edge_idx.get(uv);

        auto const group_routes = get_fbs_group_routes(e->pci_);
        auto const trip_indices = get_trip_indices(e);

        auto const edge_log =
            e->pci_ < uv.graph_log_.edge_log_.index_size()
                ? utl::to_vec(
                      uv.graph_log_.edge_log_.at(e->pci_),
                      [&](auto const& entry) {
                        return CreatePaxMonDebugEdgeLogEntry(
                            mc, to_fbs_time(sched, entry.system_time_),
                            entry.required_transfer_time_,
                            entry.available_transfer_time_,
                            static_cast<PaxMonDebugEdgeType>(entry.edge_type_),
                            entry.broken_);
                      })
                : std::vector<Offset<PaxMonDebugEdgeLogEntry>>{};

        auto const pax_log = get_fbs_pax_log(e->pci_);

        return CreatePaxMonDebugEdge(
            mc, e->from_, e->to_, edge_idx.out_edge_idx_,
            static_cast<PaxMonDebugEdgeType>(e->type_), e->broken_,
            e->is_valid(uv), e->transfer_time_,
            uv.pax_connection_info_.expected_load_.at(e->pci_),
            mc.CreateVectorOfStructs(group_routes),
            mc.CreateVector(trip_indices), mc.CreateVector(edge_log),
            mc.CreateVector(pax_log));
      });

  mc.create_and_finish(
      MsgContent_PaxMonDebugGraphResponse,
      CreatePaxMonDebugGraphResponse(
          mc, uv.graph_log_.enabled_, mc.CreateVector(fbs_nodes),
          mc.CreateVector(fbs_edges), mc.CreateVector(fbs_trips))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
