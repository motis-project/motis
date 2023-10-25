#include "motis/paxforecast/revert_forecast.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <vector>

#include "cista/reflection/comparable.h"

#include "utl/enumerate.h"
#include "utl/to_set.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/dynamic_fws_multimap.h"
#include "motis/core/common/fws_graph.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/message.h"

#include "motis/paxmon/localization.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/reachability.h"
#include "motis/paxmon/temp_passenger_group.h"

using namespace motis::paxmon;
using namespace motis::module;
using namespace flatbuffers;

namespace motis::paxforecast {

namespace {

struct reroute_node {
  CISTA_COMPARABLE()
  local_group_route_index route_{};
  float probability_{};
};

struct reroute_edge {
  CISTA_COMPARABLE()
  std::uint32_t from_{};
  std::uint32_t to_{};
};

}  // namespace

// TODO(pablo): preparation time

// TODO(pablo): simple check for now, can maybe be improved later
bool can_switch(passenger_localization const& from,
                passenger_localization const& to) {
  return from.at_station_ == to.at_station_ && from.in_trip_ == to.in_trip_;
}

bool can_switch(reroute_log_localization const& from,
                reroute_log_localization const& to) {
  return from.station_id_ == to.station_id_ && from.in_trip_ == to.in_trip_ &&
         from.trip_idx_ == to.trip_idx_;
}

fws_graph<reroute_node, reroute_edge> build_reroute_graph(
    passenger_group_container const& pgc, passenger_group_index const pgi) {
  auto graph = fws_graph<reroute_node, reroute_edge>{};
  auto leaves = dynamic_fws_multimap<std::uint32_t>{};
  auto reverts = std::vector<reroute_log_entry const*>{};
  auto const routes = pgc.routes(pgi);
  auto has_reverts = false;

  graph.emplace_back_node(static_cast<local_group_route_index>(0), 1.F);
  leaves[0].emplace_back(0U);

  auto const get_parent =
      [&](std::uint32_t const node_idx) -> std::optional<std::uint32_t> {
    for (auto const& e : graph.incoming_edges(node_idx)) {
      return e.from_;
    }
    return {};
  };

  auto const process_reverts = [&]() {
    if (reverts.empty()) {
      return;
    }

    std::cout << "process_reverts: " << reverts.size() << " reverts"
              << std::endl;

    auto reactivated_routes = utl::to_set(
        reverts, [](auto const& le) { return le->old_route_.route_; });
    auto reverted_routes = std::set<local_group_route_index>{};
    auto localizations =
        std::vector<reroute_log_localization const*>(routes.size());

    for (auto const* le : reverts) {
      localizations[le->old_route_.route_] = &le->old_route_.localization_;
      for (auto const& nr : pgc.log_entry_new_routes_.at(le->index_)) {
        localizations[nr.route_] = &nr.localization_;
        if (nr.previous_probability_ > nr.new_probability_) {
          reverted_routes.insert(nr.route_);
        }
      }
    }

    std::cout << "  reactivated_routes:";
    for (auto const r : reactivated_routes) {
      std::cout << " " << r;
    }
    std::cout << "\n  reverted_routes:";
    for (auto const r : reverted_routes) {
      std::cout << " " << r;
    }
    std::cout << std::endl;

    for (auto const reverted_route : reverted_routes) {
      auto const leaf_localization = localizations.at(reverted_route);
      utl::verify(leaf_localization != nullptr,
                  "revert_forecast: leaf localization not found");

      auto const old_leaves = utl::to_vec(leaves[reverted_route],
                                          [&](auto const idx) { return idx; });
      leaves[reverted_route].clear();

      std::cout << "  reverting route " << reverted_route << ":\n";

      for (auto const old_leaf : old_leaves) {
        auto candidate = std::optional<std::uint32_t>{};  // node index

        std::cout << "    leaf node " << old_leaf << ", route "
                  << reverted_route << "\n";
        for (auto node_idx = get_parent(old_leaf); node_idx;
             node_idx = get_parent(*node_idx)) {
          auto const& node = graph.nodes_.at(*node_idx);
          std::cout << "    parent node " << *node_idx << ", route "
                    << node.route_ << "\n";
          if (!reactivated_routes.contains(node.route_)) {
            continue;
          }
          auto const candidate_localization = localizations.at(node.route_);
          utl::verify(candidate_localization != nullptr,
                      "revert_forecast: candidate localization not found");
          if (can_switch(*leaf_localization, *candidate_localization)) {
            candidate = *node_idx;
            break;
          }
        }

        if (candidate) {
          auto const candidate_node = graph.nodes_.at(*candidate);
          auto const new_node_idx = graph.nodes_.size();
          auto const old_leaf_prob = graph.nodes_.at(old_leaf).probability_;
          graph.emplace_back_node(candidate_node.route_, old_leaf_prob);
          graph.push_back_edge(reroute_edge{old_leaf, new_node_idx});
          leaves[candidate_node.route_].emplace_back(new_node_idx);
          std::cout << "    candidate found: node " << *candidate << ", route "
                    << candidate_node.route_ << ", probability "
                    << candidate_node.probability_ << " -> new node "
                    << new_node_idx << ", probability " << old_leaf_prob
                    << "\n";
        } else {
          leaves[reverted_route].emplace_back(old_leaf);
          std::cout << "    no candidate found, keeping old leaf " << old_leaf
                    << " for route " << reverted_route << "\n";
        }
      }
    }

    reverts.clear();
    std::cout << "process_reverts done" << std::endl;
  };

  for (auto const& le : pgc.reroute_log_entries(pgi)) {
    if (le.reason_ == reroute_reason_t::REVERT_FORECAST) {
      has_reverts = true;
      std::cout << "revert_forecast: multiple reverts for group " << pgi
                << std::endl;
      if (!reverts.empty() &&
          reverts.front()->update_number_ != le.update_number_) {
        process_reverts();
      }
      reverts.emplace_back(&le);
    } else {
      process_reverts();
      auto const prev_nodes = utl::to_vec(leaves[le.old_route_.route_],
                                          [&](auto const idx) { return idx; });
      leaves[le.old_route_.route_].clear();
      auto const new_routes = pgc.log_entry_new_routes_.at(le.index_);

      auto const total_outgoing_prob =
          std::accumulate(begin(new_routes), end(new_routes), 0.F,
                          [&](auto const sum, auto const& new_route) {
                            if (new_route.route_ == le.old_route_.route_) {
                              return sum + new_route.new_probability_;
                            } else {
                              return sum + (new_route.new_probability_ -
                                            new_route.previous_probability_);
                            }
                          });

      for (auto const prev_node_idx : prev_nodes) {
        auto const parent_prob = graph.nodes_[prev_node_idx].probability_;
        for (auto const& new_route : new_routes) {
          auto const prob_change = new_route.route_ == le.old_route_.route_
                                       ? new_route.new_probability_
                                       : new_route.new_probability_ -
                                             new_route.previous_probability_;
          auto const pick_prob = prob_change / total_outgoing_prob;
          auto const abs_prob = pick_prob * parent_prob;
          auto const new_node_idx = graph.nodes_.size();
          graph.emplace_back_node(new_route.route_, abs_prob);
          graph.push_back_edge(reroute_edge{prev_node_idx, new_node_idx});
          leaves[new_route.route_].emplace_back(new_node_idx);
        }
      }
    }
  }
  process_reverts();

  if (has_reverts) {
    std::cout << "graph for group " << pgi << " includes reverts:\n";
    for (auto const& [node_idx, node] : utl::enumerate(graph.nodes_)) {
      std::cout << "  node " << node_idx << ": route = " << node.route_
                << ", probability = " << node.probability_ << "\n";
      for (auto const& edge : graph.outgoing_edges(node_idx)) {
        std::cout << "    edge " << edge.from_ << " -> " << edge.to_ << "\n";
      }
    }
  }

  return graph;
}

void revert_forecast(universe& uv, schedule const& sched,
                     FlatBufferBuilder& fbb,
                     std::vector<Offset<PaxMonRerouteGroup>>& reroutes,
                     passenger_group_index const pgi,
                     std::vector<bool>& unbroken_routes) {
  auto const& pgc = uv.passenger_groups_;

  auto const graph = build_reroute_graph(pgc, pgi);
  if (unbroken_routes.size() != 1 || graph.nodes_.size() != 1) {
    std::cout << "  revert_forecast: group " << pgi << ": "
              << graph.nodes_.size() << " graph nodes, unbroken_routes: [";
    for (auto const& b : unbroken_routes) {
      std::cout << " " << b;
    }
    std::cout << " ]" << std::endl;
  }
  if (graph.nodes_.size() <= 1) {
    return;
  }

  // for localization
  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  auto const search_time =
      static_cast<time>(current_time + uv.preparation_time_);

  auto const routes = pgc.routes(pgi);
  utl::verify(routes.size() == unbroken_routes.size(),
              "revert_forecast: invalid unbroken_routes size");

  // remove routes that already have probability > 0
  // TODO(pablo): support reverting major delay reroutes
  auto unbroken_count = 0U;
  for (auto i = 0U; i < routes.size(); ++i) {
    if (unbroken_routes[i]) {
      if (routes[i].probability_ != 0.F) {
        unbroken_routes[i] = false;
      } else {
        ++unbroken_count;
      }
    }
  }
  if (unbroken_count == 0) {
    return;
  }

  auto const localizations = utl::to_vec(routes, [&](auto const& route) {
    if (route.broken_) {
      return passenger_localization{};
    } else {
      auto const reachability =
          get_reachability(uv, pgc.journey(route.compact_journey_index_));
      return localize(sched, reachability, search_time);
    }
  });

  constexpr auto const kNoParent = std::numeric_limits<std::uint32_t>::max();
  auto parents = std::vector<std::uint32_t>(graph.nodes_.size(), kNoParent);
  auto stack = std::vector<std::uint32_t>{0U};
  auto leaves = std::vector<std::uint32_t>{};

  while (!stack.empty()) {
    auto const parent_idx = stack.back();
    stack.pop_back();
    auto const outgoing_edges = graph.outgoing_edges(parent_idx);
    for (auto const& e : outgoing_edges) {
      parents[e.to_] = parent_idx;
      stack.emplace_back(e.to_);
    }
    if (outgoing_edges.empty()) {
      leaves.emplace_back(parent_idx);
    }
  }

  auto prob_changes = std::vector<float>(routes.size() * routes.size());

  for (auto const leaf_idx : leaves) {
    auto const& leaf = graph.nodes_[leaf_idx];
    auto const& leaf_loc = localizations.at(leaf.route_);
    auto candidate = std::optional<std::uint32_t>{};

    for (auto node_idx = parents[leaf_idx]; node_idx != kNoParent;
         node_idx = parents[node_idx]) {
      auto const& node = graph.nodes_[node_idx];
      if (!unbroken_routes[node.route_]) {
        continue;
      }

      auto const& node_loc = localizations.at(node.route_);
      if (!can_switch(leaf_loc, node_loc)) {
        std::cout << "revert_forecast: localization mismatch (group " << pgi
                  << ")" << std::endl;
        continue;
      }

      candidate = node.route_;
    }

    if (candidate) {
      // candidate found -> move probability from leaf to candidate
      auto const c = *candidate;
      auto const offset = c * routes.size();
      prob_changes[offset + leaf.route_] -= leaf.probability_;
      prob_changes[offset + c] += leaf.probability_;
      std::cout << "~ group " << pgi << ": leaf node " << leaf_idx << " (route "
                << leaf.route_ << ") reverting to route " << *candidate
                << ", probability " << leaf.probability_ << "\n";
    }
  }

  auto problem = false;
  auto reverted = false;
  auto new_routes = std::vector<Offset<PaxMonGroupRoute>>{};
  for (auto reactivated_route_idx = static_cast<local_group_route_index>(0);
       reactivated_route_idx < routes.size(); ++reactivated_route_idx) {
    auto const offset = reactivated_route_idx * routes.size();

    for (auto route_idx = static_cast<local_group_route_index>(0);
         route_idx < routes.size(); ++route_idx) {
      auto const p_change = prob_changes[offset + route_idx];
      if (p_change == 0.F) {
        continue;
      }
      auto const& route = routes.at(route_idx);
      auto const new_prob = route.probability_ + p_change;
      std::cout << "    reactivated route = " << reactivated_route_idx
                << ", reverted route = " << route_idx
                << ", p_change = " << p_change << ", current probs: route "
                << route_idx << " = " << route.probability_ << " => "
                << new_prob << std::endl;
      if (new_prob < -0.05F) {
        problem = true;
      }
      new_routes.emplace_back(
          to_fbs(sched, fbb,
                 temp_group_route{
                     route_idx, p_change, compact_journey{}, INVALID_TIME,
                     0 /* estimated delay - updated by reroute groups api */,
                     route_source_flags::NONE, false /* planned */
                 }));
    }

    if (!new_routes.empty()) {
      reroutes.emplace_back(CreatePaxMonRerouteGroup(
          fbb, pgi, reactivated_route_idx, fbb.CreateVector(new_routes),
          paxmon::PaxMonRerouteReason_RevertForecast,
          broken_transfer_info_to_fbs(fbb, sched, std::nullopt), false,
          fbb.CreateVector(std::vector<Offset<PaxMonLocalizationWrapper>>{
              to_fbs_localization_wrapper(
                  sched, fbb, localizations.at(reactivated_route_idx))})));
      reverted = true;
    }
    new_routes.clear();
  }

  if (problem) {
    std::cout << "revert_forecast: PROBLEM DETECTED! group " << pgi
              << std::endl;
  }

  if (problem || reverted) {
    std::cout << "revert_forecast group " << pgi << ", "
              << pgc.reroute_log_entries(pgi).size()
              << " reroute log entries:\n";
    //      message_creator mc;
    //      mc.create_and_finish(
    //          MsgContent_PaxMonGetGroupsRequest,
    //          CreatePaxMonGetGroupsRequest(
    //              mc, uv.id_,
    //              mc.CreateVector(std::vector<passenger_group_index>{pgi}),
    //              mc.CreateVector(std::vector<Offset<PaxMonDataSource>>{}),
    //              true) .Union(),
    //          "/paxmon/get_groups");
    //      auto const req = make_msg(mc);
    //      auto const res = motis_call(req)->val();
    //      std::cout << "group info:\n"
    //                << res->to_json(json_format::CONTENT_ONLY_TYPES_IN_UNIONS)
    //                << "\n\n"
    //                << std::endl;

    std::cout << "graph:\n";
    for (auto const& [node_idx, node] : utl::enumerate(graph.nodes_)) {
      std::cout << "  node " << node_idx << ": route = " << node.route_
                << ", probability = " << node.probability_ << "\n";
      for (auto const& edge : graph.outgoing_edges(node_idx)) {
        std::cout << "    edge " << edge.from_ << " -> " << edge.to_ << "\n";
      }
    }
    std::cout << "\nparents:\n";
    for (auto const& [node_idx, parent_idx] : utl::enumerate(parents)) {
      std::cout << "  node " << node_idx << ": parent = " << parent_idx << "\n";
    }
    std::cout << "\nleaves:";
    for (auto const& leaf_idx : leaves) {
      std::cout << " " << leaf_idx;
    }
    std::cout << "\n" << std::endl;

    std::cout << "probability changes:\n";
    for (auto reactivated_route_idx = static_cast<local_group_route_index>(0);
         reactivated_route_idx < routes.size(); ++reactivated_route_idx) {
      auto const offset = reactivated_route_idx * routes.size();

      for (auto route_idx = static_cast<local_group_route_index>(0);
           route_idx < routes.size(); ++route_idx) {
        auto const p_change = prob_changes[offset + route_idx];
        if (p_change == 0.F) {
          continue;
        }
        std::cout << "  route " << route_idx << " -> " << reactivated_route_idx
                  << ": " << p_change << " (current p for route " << route_idx
                  << ": " << routes.at(route_idx).probability_ << ")\n";
      }
    }
    std::cout << std::endl;
  }
}

void revert_forecasts(universe& uv, schedule const& sched,
                      std::vector<passenger_group_with_route> const& pgwrs) {
  auto const constexpr BATCH_SIZE = 5'000;
  // TODO(pablo): refactoring (update_tracked_groups)
  message_creator mc;
  auto reroutes = std::vector<Offset<PaxMonRerouteGroup>>{};

  auto const send_reroutes = [&]() {
    if (reroutes.empty()) {
      return;
    }
    std::cout << "revert_forecasts: sending " << reroutes.size() << " reroutes"
              << std::endl;
    mc.create_and_finish(
        MsgContent_PaxMonRerouteGroupsRequest,
        CreatePaxMonRerouteGroupsRequest(mc, uv.id_, mc.CreateVector(reroutes))
            .Union(),
        "/paxmon/reroute_groups");
    auto const msg = make_msg(mc);
    motis_call(msg)->val();
    reroutes.clear();
    mc.Clear();
  };

  std::cout
      << "=================================================================\n"
      << "revert_forecasts: " << pgwrs.size() << " groups" << std::endl;

  auto const pgc = uv.passenger_groups_;
  auto current_pgi = std::numeric_limits<passenger_group_index>::max();
  auto unbroken_routes = std::vector<bool>{};

  auto const handle_group = [&]() {
    if (current_pgi == std::numeric_limits<passenger_group_index>::max()) {
      return;
    }
    revert_forecast(uv, sched, mc, reroutes, current_pgi, unbroken_routes);
    if (reroutes.size() >= BATCH_SIZE) {
      send_reroutes();
    }
  };

  for (auto const& pgwr : pgwrs) {
    if (pgwr.pg_ != current_pgi) {
      handle_group();
      current_pgi = pgwr.pg_;
      unbroken_routes.clear();
      unbroken_routes.resize(pgc.routes(pgwr.pg_).size());
    }
    unbroken_routes[pgwr.route_] = true;
  }

  handle_group();

  send_reroutes();
}

}  // namespace motis::paxforecast
