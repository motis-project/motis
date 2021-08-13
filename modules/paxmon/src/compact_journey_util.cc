#include "motis/paxmon/compact_journey_util.h"

#include <algorithm>
#include <iterator>

#include "utl/verify.h"

#include "motis/core/access/realtime_access.h"
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
    auto const sections = access::sections(leg.trip_);
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
  if (last_prefix_leg.trip_ == first_suffix_leg.trip_) {
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
                          new_first_suffix_leg.enter_station_id_);
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

}  // namespace motis::paxmon
