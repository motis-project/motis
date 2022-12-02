#include "motis/paxforecast/revert_forecast.h"

#include <cstdint>
#include <iostream>
#include <limits>

#include "utl/enumerate.h"
#include "utl/to_vec.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/message.h"

#include "motis/paxmon/messages.h"
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
    out << "{old_route=" << print_log_route_info{e.old_route_} << ", reason=";
    switch (e.reason_) {
      case reroute_reason_t::MANUAL: out << "MANUAL"; break;
      case reroute_reason_t::BROKEN_TRANSFER: out << "BROKEN_TRANSFER"; break;
      case reroute_reason_t::MAJOR_DELAY_EXPECTED:
        out << "MAJOR_DELAY_EXPECTED";
        break;
      case reroute_reason_t::REVERT_FORECAST: out << "REVERT_FORECAST"; break;
      case reroute_reason_t::SIMULATION: out << "SIMULATION"; break;
      case reroute_reason_t::UPDATE_FORECAST: out << "UPDATE_FORECAST"; break;
    }
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

}  // namespace

void revert_forecast(universe& uv, schedule const& sched,
                     FlatBufferBuilder& fbb,
                     std::vector<Offset<PaxMonRerouteGroup>>& reroutes,
                     passenger_group_with_route const& pgwr) {
  auto const& pgc = uv.passenger_groups_;
  auto const log_entries = pgc.reroute_log_entries(pgwr.pg_);
  if (log_entries.empty()) {
    return;
  }

  std::cout << "revert_forecast: pg=" << pgwr.pg_ << ", route=" << pgwr.route_
            << ", log_entries=" << log_entries.size() << "\n";

  for (auto const& entry : log_entries) {
    std::cout << "  " << print_log_entry{entry, pgc} << "\n";
  }

  auto const routes = pgc.routes(pgwr.pg_);
  auto orig_probs = utl::to_vec(
      routes, [](group_route const& gr) { return gr.probability_; });

  auto const print_probs = [&]() {
    std::cout << "  --> route probs=[";
    for (auto const& p : orig_probs) {
      std::cout << " " << p;
    }
    std::cout << " ]\n";
  };

  print_probs();
  for (auto idx_to_revert = log_entries.size() - 1;; --idx_to_revert) {
    auto const& entry = log_entries.at(idx_to_revert);
    std::cout << "  log entry " << idx_to_revert << ": "
              << print_log_entry{entry, pgc} << "\n";
    orig_probs[entry.old_route_.route_] =
        entry.old_route_.previous_probability_;
    for (auto const& nr : pgc.log_entry_new_routes_.at(entry.index_)) {
      orig_probs[nr.route_] = nr.previous_probability_;
    }
    print_probs();
    if (entry.old_route_.route_ == pgwr.route_) {
      if (entry.reason_ == reroute_reason_t::REVERT_FORECAST ||
          entry.reason_ == reroute_reason_t::MANUAL) {
        std::cout << "  abort: because of " << print_log_entry{entry, pgc}
                  << std::endl;
        return;
      } else {
        break;
      }
    }
    if (idx_to_revert == 0) {
      std::cout << "  abort: no matching reroute entry found" << std::endl;
      return;
    }
  }
  auto new_routes = std::vector<Offset<PaxMonGroupRoute>>{};
  for (auto const [i, p] : utl::enumerate(orig_probs)) {
    new_routes.emplace_back(
        to_fbs(sched, fbb,
               temp_group_route{
                   static_cast<local_group_route_index>(i), p,
                   compact_journey{}, INVALID_TIME,
                   0 /* estimated delay - updated by reroute groups api */,
                   route_source_flags::NONE, false /* planned */
               }));
  }
  reroutes.emplace_back(CreatePaxMonRerouteGroup(
      fbb, pgwr.pg_, pgwr.route_, fbb.CreateVector(new_routes),
      paxmon::PaxMonRerouteReason_RevertForecast,
      broken_transfer_info_to_fbs(fbb, sched, std::nullopt), true));
  std::cout << std::endl;
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

  auto last_group = std::numeric_limits<passenger_group_index>::max();
  for (auto const& pgwr : pgwrs) {
    if (pgwr.pg_ == last_group) {
      // TODO(pablo): for now, always revert the earliest route
      continue;
    }
    last_group = pgwr.pg_;
    revert_forecast(uv, sched, mc, reroutes, pgwr);
    if (reroutes.size() >= BATCH_SIZE) {
      send_reroutes();
    }
  }

  send_reroutes();
}

}  // namespace motis::paxforecast
