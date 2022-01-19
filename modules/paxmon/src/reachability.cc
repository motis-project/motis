#include "motis/paxmon/reachability.h"

#include "utl/enumerate.h"
#include "utl/verify.h"

#include "motis/core/access/trip_access.h"

namespace motis::paxmon {

reachability_info get_reachability(universe const& uv,
                                   compact_journey const& j) {
  utl::verify(!j.legs_.empty(), "empty journey");

  auto reachability = reachability_info{};
  auto ok = true;
  auto const& first_leg = j.legs_.front();
  auto station_arrival_time = first_leg.enter_time_;

  reachability.reachable_interchange_stations_.emplace_back(
      reachable_station{first_leg.enter_station_id_, first_leg.enter_time_,
                        first_leg.enter_time_});

  for (auto const& [leg_idx, leg] : utl::enumerate(j.legs_)) {
    auto const tdi = uv.trip_data_.get_index(leg.trip_idx_);
    auto in_trip = false;
    auto entry_ok = false;
    auto exit_ok = false;
    for (auto [edge_idx, ei] : utl::enumerate(uv.trip_data_.edges(tdi))) {
      auto const* e = ei.get(uv);
      if (!in_trip) {
        auto const from = e->from(uv);
        if (from->station_idx() == leg.enter_station_id_ &&
            from->schedule_time() == leg.enter_time_) {
          auto required_arrival_time_at_station = from->current_time();
          if (leg.enter_transfer_) {
            required_arrival_time_at_station -= leg.enter_transfer_->duration_;
          }
          if (station_arrival_time > required_arrival_time_at_station ||
              from->is_canceled()) {
            ok = false;
            break;
          }
          in_trip = true;
          reachability.reachable_trips_.emplace_back(
              reachable_trip{leg.trip_idx_, tdi, &leg, from->schedule_time(),
                             INVALID_TIME, from->current_time(), INVALID_TIME,
                             edge_idx, reachable_trip::INVALID_INDEX});
          entry_ok = true;
        }
      }
      if (in_trip) {
        auto const to = e->to(uv);
        if (to->schedule_time() > leg.exit_time_) {
          ok = false;
          break;
        }
        if (to->station_idx() == leg.exit_station_id_ &&
            to->schedule_time() == leg.exit_time_) {
          if (to->is_canceled()) {
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
      auto const is_last_leg = leg_idx == j.legs_.size() - 1;
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
