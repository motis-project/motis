#include "motis/paxforecast/revert_forecast.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include "cista/reflection/comparable.h"

#include "utl/enumerate.h"
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

struct print_log_route_info {
  friend std::ostream& operator<<(std::ostream& out,
                                  print_log_route_info const& p) {
    auto const& ri = p.ri_;
    out << "{r=" << ri.route_ << ", p=" << ri.previous_probability_ << "->"
        << ri.new_probability_ << "}";
    return out;
  }
  reroute_log_route_info const& ri_;
};

struct print_log_entry {
  friend std::ostream& operator<<(std::ostream& out, print_log_entry const& p) {
    auto const& e = p.entry_;
    out << "{old_route=" << print_log_route_info{e.old_route_}
        << ", reason=" << e.reason_;
    auto const new_routes = p.pgc_.log_entry_new_routes_.at(e.index_);
    out << ", new_routes=[";
    for (auto const& nr : new_routes) {
      out << " " << print_log_route_info{nr};
    }
    out << " ]";
    if (e.broken_transfer_) {
      auto const& t = e.broken_transfer_.value();
      out << ", broken_transfer={"
          << "leg=" << t.leg_index_ << "}";
    }
    out << "}";
    return out;
  }

  reroute_log_entry const& entry_;
  passenger_group_container const& pgc_;
};

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

fws_graph<reroute_node, reroute_edge> build_reroute_graph(
    passenger_group_container const& pgc, passenger_group_index const pgi) {
  fws_graph<reroute_node, reroute_edge> graph;
  dynamic_fws_multimap<std::uint32_t> leaves;

  graph.emplace_back_node(static_cast<local_group_route_index>(0), 1.F);
  leaves[0].emplace_back(0U);

  for (auto const& le : pgc.reroute_log_entries(pgi)) {
    if (le.reason_ == reroute_reason_t::REVERT_FORECAST) {
      std::cout << "build_reroute_graph: REVERT_FORECAST NOT YET SUPPORTED"
                << std::endl;
      graph.nodes_.clear();
      return graph;
    } else {
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

  return graph;
}

void revert_forecast(universe& uv, schedule const& sched,
                     FlatBufferBuilder& fbb,
                     std::vector<Offset<PaxMonRerouteGroup>>& reroutes,
                     passenger_group_index const pgi,
                     std::vector<bool> const& unbroken_routes) {
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
  auto const localizations = utl::to_vec(routes, [&](auto const& route) {
    if (route.broken_) {
      return passenger_localization{};
    } else {
      auto const reachability =
          get_reachability(uv, pgc.journey(route.compact_journey_index_));
      return localize(sched, reachability, search_time);
    }
  });

  auto parents = std::vector<std::uint32_t>(graph.nodes_.size());
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
    for (auto node_idx = parents[leaf_idx]; node_idx != parents[node_idx];
         node_idx = parents[node_idx]) {
      auto const& node = graph.nodes_[node_idx];
      if (!unbroken_routes[node.route_]) {
        continue;
      }

      auto const& node_loc = localizations.at(node.route_);
      // simple check for now, can maybe be improved later
      if (leaf_loc.at_station_ != node_loc.at_station_ ||
          leaf_loc.in_trip_ != node_loc.in_trip_) {
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
    }
  }

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
    }
    new_routes.clear();
  }
}

void revert_forecasts(
    universe& uv, schedule const& sched, simulation_result const& sim_result,
    std::vector<passenger_group_with_route> const& pgwrs,
    mcd::hash_map<passenger_group_with_route,
                  passenger_localization const*> const& pgwr_localizations) {
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
