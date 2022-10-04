#include "motis/paxmon/reachability.h"

#include "utl/enumerate.h"
#include "utl/verify.h"

#include "motis/core/access/trip_access.h"

namespace motis::paxmon {

reachability_info get_reachability(universe const& uv,
                                   fws_compact_journey const& j) {
  auto reachability = reachability_info{};
  auto ok = true;
  auto const legs = j.legs();
  utl::verify(!legs.empty(), "empty journey");
  auto const& first_leg = legs.front();
  auto station_arrival_time = first_leg.enter_time_;
  auto current_transfer_departure_time = INVALID_TIME;
  auto current_transfer_arrival_time = INVALID_TIME;
  auto arrival_canceled = false;
  auto departure_canceled = false;
  auto required_transfer_time = std::uint16_t{0};

  reachability.reachable_interchange_stations_.emplace_back(
      reachable_station{first_leg.enter_station_id_, first_leg.enter_time_,
                        first_leg.enter_time_});

  for (auto const& [leg_idx, leg] : utl::enumerate(legs)) {
    auto const tdi = uv.trip_data_.get_index(leg.trip_idx_);
    auto in_trip = false;
    auto entry_ok = false;
    auto exit_ok = false;
    for (auto [edge_idx, ei] : utl::enumerate(uv.trip_data_.edges(tdi))) {
      auto const* e = ei.get(uv);
      current_transfer_departure_time = INVALID_TIME;
      if (!in_trip) {
        auto const from = e->from(uv);
        if (from->station_idx() == leg.enter_station_id_ &&
            from->schedule_time() == leg.enter_time_) {
          current_transfer_departure_time = from->current_time();
          auto required_arrival_time_at_station = from->current_time();
          if (leg.enter_transfer_) {
            required_transfer_time = leg.enter_transfer_->duration_;
            required_arrival_time_at_station -= required_transfer_time;
          }
          if (station_arrival_time > required_arrival_time_at_station ||
              from->is_canceled()) {
            departure_canceled = from->is_canceled();
            ok = false;
            break;
          }
          in_trip = true;
          reachability.reachable_trips_.emplace_back(
              reachable_trip{leg.trip_idx_, tdi, leg, from->schedule_time(),
                             INVALID_TIME, from->current_time(), INVALID_TIME,
                             edge_idx, reachable_trip::INVALID_INDEX});
          entry_ok = true;
        }
      }
      if (in_trip) {
        auto const to = e->to(uv);
        if (to->schedule_time() > leg.exit_time_) {
          arrival_canceled = true;
          ok = false;
          break;
        }
        if (to->station_idx() == leg.exit_station_id_ &&
            to->schedule_time() == leg.exit_time_) {
          current_transfer_arrival_time = to->current_time();
          if (to->is_canceled()) {
            arrival_canceled = true;
            required_transfer_time = 0;
            ok = false;
            break;
          }
          station_arrival_time = to->current_time();
          auto& rt = reachability.reachable_trips_.back();
          rt.exit_real_time_ = station_arrival_time;
          rt.exit_schedule_time_ = to->schedule_time();
          rt.exit_edge_idx_ = edge_idx;
          reachability.reachable_interchange_stations_.emplace_back(
              reachable_station{to->station_, to->schedule_time_,
                                station_arrival_time});
          exit_ok = true;
          break;
        }
      }
    }
    if (!entry_ok || !exit_ok) {
      ok = false;
    }
    if (!ok) {
      auto const is_first_leg = leg_idx == 0;
      auto const is_last_leg = leg_idx == legs.size() - 1;
      if (current_transfer_arrival_time == INVALID_TIME) {
        arrival_canceled = true;
      }
      if (current_transfer_departure_time == INVALID_TIME) {
        departure_canceled = true;
      }
      reachability.first_unreachable_transfer_ = broken_transfer_info{
          static_cast<std::uint16_t>(leg_idx),
          !entry_ok ? transfer_direction_t::ENTER : transfer_direction_t::EXIT,
          current_transfer_arrival_time,
          current_transfer_departure_time,
          required_transfer_time,
          arrival_canceled,
          departure_canceled};
      if (!entry_ok) {
        if (is_first_leg) {
          reachability.status_ = reachability_status::BROKEN_INITIAL_ENTRY;
        } else {
          reachability.status_ = reachability_status::BROKEN_TRANSFER_ENTRY;
        }
      } else if (!exit_ok) {
        if (is_last_leg) {
          reachability.status_ = reachability_status::BROKEN_FINAL_EXIT;
        } else {
          reachability.status_ = reachability_status::BROKEN_TRANSFER_EXIT;
        }
      }
      break;
    }
  }

  reachability.ok_ = ok;
  return reachability;
}

}  // namespace motis::paxmon
