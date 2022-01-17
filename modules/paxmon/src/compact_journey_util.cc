#include "motis/paxmon/compact_journey_util.h"

#include <algorithm>
#include <iterator>

#include "utl/verify.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"

#include "motis/paxmon/checks.h"
#include "motis/paxmon/debug.h"
#include "motis/paxmon/util/interchange_time.h"

using namespace motis::paxmon::util;

namespace motis::paxmon {

compact_journey get_prefix(schedule const& sched, compact_journey const& cj,
                           passenger_localization const& loc) {
  auto prefix = compact_journey{};

  if (loc.first_station_) {
    return prefix;
  }

  for (auto const& leg : cj.legs_) {
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
    auto& new_leg = prefix.legs_.emplace_back(leg);
    if (exit_section_it != end(sections)) {
      auto const exit_section = *exit_section_it;
      new_leg.exit_station_id_ = exit_section.to_station_id();
      new_leg.exit_time_ = get_schedule_time(sched, exit_section.ev_key_to());
      break;
    }
  }

  return prefix;
}

std::pair<compact_journey, time> get_prefix_and_arrival_time(
    schedule const& sched, compact_journey const& cj,
    unsigned const search_station, time const earliest_arrival) {
  auto prefix = compact_journey{};
  auto current_arrival_time = INVALID_TIME;

  for (auto const& leg : cj.legs_) {
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
        auto& new_leg = prefix.legs_.emplace_back(leg);
        new_leg.exit_station_id_ = search_station;
        new_leg.exit_time_ =
            get_schedule_time(sched, search_section.ev_key_to());
        current_arrival_time = search_section.lcon().a_time_;
      }
      break;
    } else {
      prefix.legs_.emplace_back(leg);
    }
  }

  return {prefix, current_arrival_time};
}

compact_journey get_suffix(schedule const& sched, compact_journey const& cj,
                           passenger_localization const& loc) {
  if (loc.first_station_) {
    return cj;
  }

  auto suffix = compact_journey{};

  if (loc.in_trip()) {
    auto in_trip = false;
    for (auto const& leg : cj.legs_) {
      if (in_trip) {
        suffix.legs_.emplace_back(leg);
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
          auto& new_leg = suffix.legs_.emplace_back(leg);
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
    for (auto const& leg : cj.legs_) {
      if (!in_trip) {
        if (leg.enter_station_id_ == loc_station &&
            leg.enter_time_ >= loc.schedule_arrival_time_) {
          in_trip = true;
        } else {
          continue;
        }
      }
      suffix.legs_.emplace_back(leg);
    }
  }

  return suffix;
}

compact_journey merge_journeys(schedule const& sched,
                               compact_journey const& prefix,
                               compact_journey const& suffix) {
  if (prefix.legs_.empty()) {
    return suffix;
  } else if (suffix.legs_.empty()) {
    return prefix;
  }

  auto merged = prefix;
  auto const& last_prefix_leg = prefix.legs_.back();
  auto const& first_suffix_leg = suffix.legs_.front();
  if (last_prefix_leg.trip_idx_ == first_suffix_leg.trip_idx_) {
    auto& merged_leg = merged.legs_.back();
    merged_leg.exit_station_id_ = first_suffix_leg.exit_station_id_;
    merged_leg.exit_time_ = first_suffix_leg.exit_time_;
    std::copy(std::next(begin(suffix.legs_)), end(suffix.legs_),
              std::back_inserter(merged.legs_));
  } else {
    std::copy(begin(suffix.legs_), end(suffix.legs_),
              std::back_inserter(merged.legs_));
    auto& new_first_suffix_leg = merged.legs_[prefix.legs_.size()];
    new_first_suffix_leg.enter_transfer_ =
        get_transfer_info(sched, last_prefix_leg.exit_station_id_,
                          get_arrival_track(sched, last_prefix_leg),
                          new_first_suffix_leg.enter_station_id_,
                          get_departure_track(sched, new_first_suffix_leg));
  }

  /*
  if (!check_compact_journey(sched, merged)) {
    std::cout << "\nprefix journey:\n";
    for (auto const& leg : prefix.legs_) {
      print_leg(sched, leg);
    }
    std::cout << "\nsuffix journey:\n";
    for (auto const& leg : suffix.legs_) {
      print_leg(sched, leg);
    }

    throw utl::fail("merge_journeys: invalid result");
  }
  */

  return merged;
}

inline bool is_long_distance_class(service_class const clasz) {
  return clasz >= service_class::ICE && clasz <= service_class::N;
}

std::optional<unsigned> get_first_long_distance_station_id(
    universe const& uv, compact_journey const& cj) {
  for (auto const& leg : cj.legs_) {
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

std::optional<unsigned> get_last_long_distance_station_id(
    universe const& uv, compact_journey const& cj) {
  for (auto it = std::rbegin(cj.legs_); it != std::rend(cj.legs_); ++it) {
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

std::optional<std::uint16_t> get_arrival_track(schedule const& sched,
                                               trip const* trp,
                                               std::uint32_t exit_station_id,
                                               motis::time exit_time) {
  if (trp == nullptr) {
    return {};
  }
  auto const sections = access::sections(trp);
  auto const section_it = std::find_if(
      begin(sections), end(sections), [&](access::trip_section const& sec) {
        return sec.to_station_id() == exit_station_id &&
               get_schedule_time(sched, sec.ev_key_to()) == exit_time;
      });
  if (section_it != end(sections)) {
    return (*section_it).lcon().full_con_->a_track_;
  } else {
    return {};
  }
}

std::optional<std::uint16_t> get_arrival_track(schedule const& sched,
                                               journey_leg const& leg) {
  return get_arrival_track(sched, get_trip(sched, leg.trip_idx_),
                           leg.exit_station_id_, leg.exit_time_);
}

std::optional<std::uint16_t> get_departure_track(schedule const& sched,
                                                 trip const* trp,
                                                 std::uint32_t enter_station_id,
                                                 motis::time enter_time) {
  if (trp == nullptr) {
    return {};
  }
  auto const sections = access::sections(trp);
  auto const section_it = std::find_if(
      begin(sections), end(sections), [&](access::trip_section const& sec) {
        return sec.from_station_id() == enter_station_id &&
               get_schedule_time(sched, sec.ev_key_from()) == enter_time;
      });
  if (section_it != end(sections)) {
    return (*section_it).lcon().full_con_->d_track_;
  } else {
    return {};
  }
}

std::optional<std::uint16_t> get_departure_track(schedule const& sched,
                                                 journey_leg const& leg) {
  return get_departure_track(sched, get_trip(sched, leg.trip_idx_),
                             leg.enter_station_id_, leg.enter_time_);
}

}  // namespace motis::paxmon
