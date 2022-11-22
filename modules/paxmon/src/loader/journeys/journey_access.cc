#include "motis/paxmon/loader/journeys/journey_access.h"

#include "utl/enumerate.h"
#include "utl/verify.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/track_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/access/trip_section.h"

#include "motis/paxmon/util/interchange_time.h"

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

struct first_and_last_trip_section {
  std::optional<access::trip_section> first_section_;
  std::optional<access::trip_section> last_section_;
};

first_and_last_trip_section get_first_and_last_trip_section(
    schedule const& sched, trip const* trp, journey::stop const& first_stop,
    journey::stop const& last_stop) {
  auto result = first_and_last_trip_section{};

  auto const from_station_id = get_station(sched, first_stop.eva_no_)->index_;
  auto const to_station_id = get_station(sched, last_stop.eva_no_)->index_;
  auto const enter_time = unix_to_motistime(
      sched.schedule_begin_, first_stop.departure_.schedule_timestamp_);
  auto const exit_time = unix_to_motistime(
      sched.schedule_begin_, last_stop.arrival_.schedule_timestamp_);

  for (auto const& sec : access::sections(trp)) {
    if (sec.from_station_id() == from_station_id &&
        get_schedule_time(sched, sec.ev_key_from()) == enter_time) {
      result.first_section_ = sec;
    }
    if (sec.to_station_id() == to_station_id &&
        get_schedule_time(sched, sec.ev_key_to()) == exit_time) {
      result.last_section_ = sec;
      if (result.first_section_) {
        // both sections found
        break;
      }
    }
  }
  utl::verify(result.first_section_ && result.last_section_,
              "get_first_and_last_trip_section: found first={}, last={}",
              result.first_section_.has_value(),
              result.last_section_.has_value());
  return result;
}

void for_each_trip(
    journey const& j, schedule const& sched,
    std::function<void(trip const*, journey::stop const&, journey::stop const&,
                       std::optional<transfer_info> const&)> const& cb) {
  std::optional<std::size_t> enter_stop_idx, exit_stop_idx;
  std::optional<transfer_info> enter_transfer;
  std::optional<access::trip_section> arrival_section;
  for (auto const& [stop_idx, stop] : utl::enumerate(j.stops_)) {
    if (stop.exit_) {
      exit_stop_idx = stop_idx;
      utl::verify(enter_stop_idx.has_value(),
                  "invalid journey: exit stop without preceding enter stop");
      auto const trip_segments =
          get_journey_trip_segments(j, *enter_stop_idx, *exit_stop_idx);
      for (auto const& jts : trip_segments) {
        auto const* trp = get_trip(sched, jts.trip_->extern_trip_);
        auto const& from_stop = j.stops_.at(jts.from_);
        auto const& to_stop = j.stops_.at(jts.to_);
        auto const trp_sections =
            get_first_and_last_trip_section(sched, trp, from_stop, to_stop);
        auto const transfer =
            jts.from_ == *enter_stop_idx
                ? enter_transfer
                : ((arrival_section && trp_sections.first_section_)
                       ? util::get_transfer_info(sched, *arrival_section,
                                                 *trp_sections.first_section_)
                       : std::nullopt);
        arrival_section = trp_sections.last_section_;
        cb(trp, from_stop, to_stop, transfer);
      }
      enter_stop_idx.reset();
    }
    if (stop.enter_) {
      enter_stop_idx = stop_idx;
      if (exit_stop_idx) {
        auto const& exit_stop = j.stops_.at(exit_stop_idx.value());
        auto const exit_station = get_station(sched, exit_stop.eva_no_);
        auto const enter_station = get_station(sched, stop.eva_no_);
        auto const arrival_track = get_track_index(sched, stop.arrival_.track_);
        auto const departure_track =
            get_track_index(sched, stop.departure_.track_);
        enter_transfer =
            util::get_transfer_info(sched, exit_station->index_, arrival_track,
                                    enter_station->index_, departure_track);
        if (!enter_transfer) {
          auto const walk_duration =
              (stop.arrival_.schedule_timestamp_ -
               exit_stop.departure_.schedule_timestamp_) /
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
