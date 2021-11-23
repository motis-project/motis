#include "motis/paxmon/loader/journeys/journey_access.h"

#include "utl/enumerate.h"
#include "utl/verify.h"

#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_access.h"

namespace motis::paxmon {

std::vector<journey::trip const*> get_journey_trips(journey const& j,
                                                    std::size_t enter_stop_idx,
                                                    std::size_t exit_stop_idx) {
  auto trips = std::vector<journey::trip const*>{};
  for (auto const& t : j.trips_) {
    if (t.from_ >= enter_stop_idx && t.from_ < exit_stop_idx &&
        t.to_ > enter_stop_idx && t.to_ <= exit_stop_idx) {
      trips.emplace_back(&t);
    }
  }
  return trips;
}

std::optional<journey_trip_segment> get_longest_journey_trip_segment(
    std::vector<journey::trip const*> const& trips,
    std::size_t start_stop_idx) {
  auto longest = std::optional<journey_trip_segment>{};
  for (auto const* jt : trips) {
    if (jt->to_ <= start_stop_idx || jt->from_ > start_stop_idx) {
      continue;
    }
    if (!longest.has_value() || longest->to_ < jt->to_) {
      longest = {jt, start_stop_idx, jt->to_};
    }
  }
  return longest;
}

std::vector<journey_trip_segment> get_journey_trip_segments(
    journey const& j, std::size_t enter_stop_idx, std::size_t exit_stop_idx) {
  auto segments = std::vector<journey_trip_segment>{};
  auto const trips = get_journey_trips(j, enter_stop_idx, exit_stop_idx);
  utl::verify(!trips.empty(), "get_journey_trips returned empty result");
  for (auto start_stop_idx = enter_stop_idx; start_stop_idx < exit_stop_idx;) {
    auto const longest_trip =
        get_longest_journey_trip_segment(trips, start_stop_idx);
    utl::verify(longest_trip.has_value(),
                "get_journey_trip_segments: did not find trip cover");
    utl::verify(segments.empty() || segments.back().to_ == longest_trip->from_,
                "get_journey_trip_segments: invalid trip cover (1)");
    segments.emplace_back(*longest_trip);
    start_stop_idx = longest_trip->to_;
  }
  utl::verify(!segments.empty() && segments.front().from_ == enter_stop_idx &&
                  segments.back().to_ == exit_stop_idx,
              "get_journey_trip_segments: invalid trip cover (2)");
  return segments;
}

void for_each_trip(
    journey const& j, schedule const& sched,
    std::function<void(trip const*, journey::stop const&, journey::stop const&,
                       std::optional<transfer_info> const&)> const& cb) {
  std::optional<std::size_t> enter_stop_idx, exit_stop_idx;
  std::optional<transfer_info> enter_transfer;
  for (auto const& [stop_idx, stop] : utl::enumerate(j.stops_)) {
    if (stop.exit_) {
      exit_stop_idx = stop_idx;
      utl::verify(enter_stop_idx.has_value(),
                  "invalid journey: exit stop without preceding enter stop");
      auto const trip_segments =
          get_journey_trip_segments(j, *enter_stop_idx, *exit_stop_idx);
      for (auto const& jts : trip_segments) {
        cb(get_trip(sched, jts.trip_->extern_trip_), j.stops_.at(jts.from_),
           j.stops_.at(jts.to_),
           jts.from_ == *enter_stop_idx ? enter_transfer : std::nullopt);
      }
      enter_stop_idx.reset();
    }
    if (stop.enter_) {
      enter_stop_idx = stop_idx;
      if (exit_stop_idx) {
        if (*exit_stop_idx == stop_idx) {
          auto const st = get_station(sched, stop.eva_no_);
          enter_transfer =
              transfer_info{static_cast<duration>(st->transfer_time_),
                            transfer_info::type::SAME_STATION};
        } else {
          auto const walk_duration =
              (stop.arrival_.schedule_timestamp_ -
               j.stops_.at(*exit_stop_idx).departure_.schedule_timestamp_) /
              60;
          enter_transfer = transfer_info{static_cast<duration>(walk_duration),
                                         transfer_info::type::FOOTPATH};
        }
      } else {
        enter_transfer.reset();
      }
      exit_stop_idx.reset();
    }
  }
}

}  // namespace motis::paxmon
