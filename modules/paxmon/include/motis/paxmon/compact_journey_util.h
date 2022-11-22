#pragma once

#include <cstdint>
#include <algorithm>
#include <iterator>
#include <optional>
#include <utility>

#include "utl/verify.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/access/trip_section.h"

#include "motis/paxmon/checks.h"
#include "motis/paxmon/compact_journey.h"
#include "motis/paxmon/debug.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/universe.h"
#include "motis/paxmon/util/interchange_time.h"

namespace motis::paxmon {

template <typename CompactJourney>
inline compact_journey get_prefix(schedule const& sched,
                                  CompactJourney const& cj,
                                  passenger_localization const& loc) {
  auto prefix = compact_journey{};

  if (loc.first_station_) {
    return prefix;
  }

  for (auto const& leg : cj.legs()) {
    auto const sections = access::sections(get_trip(sched, leg.trip_idx_));
    auto const enter_section_it = std::find_if(
        begin(sections), end(sections), [&](access::trip_section const& sec) {
          return sec.from_station_id() == leg.enter_station_id_ &&
                 get_schedule_time(sched, sec.ev_key_from()) == leg.enter_time_;
        });
    if (enter_section_it == end(sections) ||
        (*enter_section_it).lcon().d_time_ >= loc.current_arrival_time_) {
      break;
    }
    auto const exit_section_it = std::find_if(
        begin(sections), end(sections), [&](access::trip_section const& sec) {
          return sec.to_station_id() == loc.at_station_->index_ &&
                 get_schedule_time(sched, sec.ev_key_to()) ==
                     loc.schedule_arrival_time_;
        });
    auto& new_leg = prefix.legs().emplace_back(leg);
    if (exit_section_it != end(sections)) {
      auto const exit_section = *exit_section_it;
      new_leg.exit_station_id_ = exit_section.to_station_id();
      new_leg.exit_time_ = get_schedule_time(sched, exit_section.ev_key_to());
      break;
    }
  }

  return prefix;
}

template <typename CompactJourney>
inline std::pair<compact_journey, time> get_prefix_and_arrival_time(
    schedule const& sched, CompactJourney const& cj,
    unsigned const search_station, time const earliest_arrival) {
  auto prefix = compact_journey{};
  auto current_arrival_time = INVALID_TIME;

  for (auto const& leg : cj.legs()) {
    auto const sections = access::sections(get_trip(sched, leg.trip_idx_));
    auto const search_section_it = std::find_if(
        begin(sections), end(sections), [&](access::trip_section const& sec) {
          return (sec.to_station_id() == search_station &&
                  sec.ev_key_to().get_time() >= earliest_arrival) ||
                 (sec.from_station_id() == search_station &&
                  sec.ev_key_from().get_time() >= earliest_arrival);
        });
    if (search_section_it != end(sections)) {
      auto const search_section = *search_section_it;
      if (search_section.to_station_id() == search_station) {
        auto& new_leg = prefix.legs().emplace_back(leg);
        new_leg.exit_station_id_ = search_station;
        new_leg.exit_time_ =
            get_schedule_time(sched, search_section.ev_key_to());
        current_arrival_time = search_section.lcon().a_time_;
      }
      break;
    } else {
      prefix.legs().emplace_back(leg);
    }
  }

  return {prefix, current_arrival_time};
}

template <typename CompactJourney>
inline compact_journey get_suffix(schedule const& sched,
                                  CompactJourney const& cj,
                                  passenger_localization const& loc) {
  if (loc.first_station_) {
    return to_compact_journey(cj);
  }

  auto suffix = compact_journey{};

  if (loc.in_trip()) {
    auto in_trip = false;
    for (auto const& leg : cj.legs()) {
      if (in_trip) {
        suffix.legs().emplace_back(leg);
      } else if (get_trip(sched, leg.trip_idx_) == loc.in_trip_) {
        in_trip = true;
        auto const sections = access::sections(loc.in_trip_);
        auto arrival_section_it = std::find_if(
            begin(sections), end(sections),
            [&](access::trip_section const& sec) {
              return sec.to_station_id() == loc.at_station_->index_ &&
                     get_schedule_time(sched, sec.ev_key_to()) ==
                         loc.schedule_arrival_time_;
            });
        utl::verify(arrival_section_it != end(sections),
                    "get_suffix: arrival section not found");
        auto first_section_it = std::next(arrival_section_it);
        if (first_section_it != end(sections)) {
          auto& new_leg = suffix.legs().emplace_back(leg);
          auto const first_section = *first_section_it;
          new_leg.enter_station_id_ = first_section.from_station_id();
          new_leg.enter_time_ =
              get_schedule_time(sched, first_section.ev_key_from());
        }
      }
    }
  } else {
    auto const loc_station = loc.at_station_->index_;
    auto in_trip = false;
    for (auto const& leg : cj.legs()) {
      if (!in_trip) {
        if (leg.enter_station_id_ == loc_station &&
            leg.enter_time_ >= loc.schedule_arrival_time_) {
          in_trip = true;
        } else {
          continue;
        }
      }
      suffix.legs().emplace_back(leg);
    }
  }

  return suffix;
}

template <typename PrefixCompactJourney, typename SuffixCompactJourney>
inline std::optional<trip_idx_t> get_continuous_trip(
    schedule const& sched, PrefixCompactJourney const& prefix,
    SuffixCompactJourney const& suffix) {
  auto const& last_prefix_leg = prefix.legs().back();
  auto const& first_suffix_leg = suffix.legs().front();

  if (last_prefix_leg.trip_idx_ == first_suffix_leg.trip_idx_) {
    return {last_prefix_leg.trip_idx_};
  }

  auto const* prefix_trp = get_trip(sched, last_prefix_leg.trip_idx_);
  auto const* suffix_trp = get_trip(sched, first_suffix_leg.trip_idx_);
  auto const prefix_sections = access::sections(prefix_trp);
  auto const suffix_sections = access::sections(suffix_trp);

  auto const prefix_entry_section = std::find_if(
      begin(prefix_sections), end(prefix_sections),
      [&](access::trip_section const& sec) {
        return sec.from_station_id() == last_prefix_leg.enter_station_id_ &&
               get_schedule_time(sched, sec.ev_key_from()) ==
                   last_prefix_leg.enter_time_;
      });
  utl::verify(prefix_entry_section != end(prefix_sections),
              "get_continuous_trip: prefix entry section "
              "not found in trip");
  auto const suffix_exit_section = std::find_if(
      begin(suffix_sections), end(suffix_sections),
      [&](access::trip_section const& sec) {
        return sec.to_station_id() == first_suffix_leg.exit_station_id_ &&
               get_schedule_time(sched, sec.ev_key_to()) ==
                   first_suffix_leg.exit_time_;
      });
  utl::verify(suffix_exit_section != end(suffix_sections),
              "get_continuous_trip: suffix exit section "
              "not found in trip");

  auto const& prefix_trips =
      *sched.merged_trips_.at((*prefix_entry_section).lcon().trips_);
  auto const& suffix_trips =
      *sched.merged_trips_.at((*suffix_exit_section).lcon().trips_);

  for (auto const& trp : prefix_trips) {
    if (std::find(begin(suffix_trips), end(suffix_trips), trp) !=
        end(suffix_trips)) {
      return {trp->trip_idx_};
    }
  }

  return {};
}

template <typename PrefixCompactJourney, typename SuffixCompactJourney>
inline std::optional<transfer_info> get_merged_transfer_info(
    schedule const& sched, PrefixCompactJourney const& prefix,
    SuffixCompactJourney const& suffix) {
  auto const& last_prefix_leg = prefix.legs().back();
  auto const& first_suffix_leg = suffix.legs().front();

  if (last_prefix_leg.trip_idx_ == first_suffix_leg.trip_idx_) {
    // already handled by get_continuous_trip
    throw utl::fail("get_merged_transfer_info: same trip");
  }

  auto const* prefix_trp = get_trip(sched, last_prefix_leg.trip_idx_);
  auto const* suffix_trp = get_trip(sched, first_suffix_leg.trip_idx_);
  auto const prefix_sections = access::sections(prefix_trp);
  auto const suffix_sections = access::sections(suffix_trp);

  auto const last_prefix_section = std::find_if(
      begin(prefix_sections), end(prefix_sections),
      [&](access::trip_section const& sec) {
        return sec.to_station_id() == last_prefix_leg.exit_station_id_ &&
               get_schedule_time(sched, sec.ev_key_to()) ==
                   last_prefix_leg.exit_time_;
      });
  utl::verify(last_prefix_section != end(prefix_sections),
              "get_merged_transfer_info: last prefix section "
              "not found in trip");
  auto const first_suffix_section = std::find_if(
      begin(suffix_sections), end(suffix_sections),
      [&](access::trip_section const& sec) {
        return sec.from_station_id() == first_suffix_leg.enter_station_id_ &&
               get_schedule_time(sched, sec.ev_key_from()) ==
                   first_suffix_leg.enter_time_;
      });
  utl::verify(first_suffix_section != end(suffix_sections),
              "get_merged_transfer_info: first suffix section "
              "not found in trip");

  return util::get_transfer_info(sched, *last_prefix_section,
                                 *first_suffix_section);
}

template <typename PrefixCompactJourney, typename SuffixCompactJourney>
inline compact_journey merge_journeys(schedule const& sched,
                                      PrefixCompactJourney const& prefix,
                                      SuffixCompactJourney const& suffix) {
  if (prefix.legs().empty()) {
    return suffix;
  } else if (suffix.legs().empty()) {
    return prefix;
  }

  auto merged = prefix;
  auto const& first_suffix_leg = suffix.legs().front();
  auto const continuous_trip_idx = get_continuous_trip(sched, prefix, suffix);
  if (continuous_trip_idx) {
    auto& merged_leg = merged.legs().back();
    merged_leg.trip_idx_ = *continuous_trip_idx;
    merged_leg.exit_station_id_ = first_suffix_leg.exit_station_id_;
    merged_leg.exit_time_ = first_suffix_leg.exit_time_;
    std::copy(std::next(begin(suffix.legs_)), end(suffix.legs_),
              std::back_inserter(merged.legs_));
  } else {
    std::copy(begin(suffix.legs_), end(suffix.legs_),
              std::back_inserter(merged.legs_));
    auto& new_first_suffix_leg = merged.legs_[prefix.legs_.size()];
    new_first_suffix_leg.enter_transfer_ =
        get_merged_transfer_info(sched, prefix, suffix);
  }

  return merged;
}

inline bool is_long_distance_class(service_class const clasz) {
  return clasz >= service_class::ICE && clasz <= service_class::N;
}

template <typename CompactJourney>
std::optional<unsigned> get_first_long_distance_station_id(
    universe const& uv, CompactJourney const& cj) {
  for (auto const& leg : cj.legs()) {
    auto const tdi = uv.trip_data_.get_index(leg.trip_idx_);
    for (auto const ei : uv.trip_data_.edges(tdi)) {
      auto const* e = ei.get(uv);
      auto const* from = e->from(uv);
      if (from->station_idx() == leg.enter_station_id_ &&
          from->schedule_time() == leg.enter_time_) {
        if (is_long_distance_class(e->clasz_)) {
          return {leg.enter_station_id_};
        }
        break;
      }
    }
  }
  return {};
}

template <typename CompactJourney>
std::optional<unsigned> get_last_long_distance_station_id(
    universe const& uv, CompactJourney const& cj) {
  for (auto it = std::rbegin(cj.legs()); it != std::rend(cj.legs()); ++it) {
    auto const& leg = *it;
    auto const tdi = uv.trip_data_.get_index(leg.trip_idx_);
    for (auto const ei : uv.trip_data_.edges(tdi)) {
      auto const* e = ei.get(uv);
      auto const* from = e->from(uv);
      if (from->station_idx() == leg.enter_station_id_ &&
          from->schedule_time() == leg.enter_time_) {
        if (is_long_distance_class(e->clasz_)) {
          return {leg.exit_station_id_};
        }
        break;
      }
    }
  }
  return {};
}

std::optional<access::trip_section> get_arrival_section(
    schedule const& sched, trip const* trp, std::uint32_t exit_station_id,
    motis::time exit_time);

std::optional<std::uint16_t> get_arrival_track(
    std::optional<access::trip_section> const& arrival_section);

std::optional<std::uint16_t> get_arrival_track(schedule const& sched,
                                               trip const* trp,
                                               std::uint32_t exit_station_id,
                                               motis::time exit_time);

std::optional<std::uint16_t> get_arrival_track(schedule const& sched,
                                               journey_leg const& leg);

std::optional<access::trip_section> get_departure_section(
    schedule const& sched, trip const* trp, std::uint32_t enter_station_id,
    motis::time enter_time);

std::optional<std::uint16_t> get_departure_track(
    std::optional<access::trip_section> const& departure_section);

std::optional<std::uint16_t> get_departure_track(schedule const& sched,
                                                 trip const* trp,
                                                 std::uint32_t enter_station_id,
                                                 motis::time enter_time);

std::optional<std::uint16_t> get_departure_track(schedule const& sched,
                                                 journey_leg const& leg);

}  // namespace motis::paxmon
