#include "motis/paxmon/service_info.h"

#include <algorithm>

#include "utl/to_vec.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/service_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/hash_map.h"

namespace motis::paxmon {

service_info get_service_info(schedule const& sched, connection const& fc,
                              connection_info const* ci) {
  return service_info{get_service_name(sched, ci),
                      sched.categories_.at(ci->family_)->name_.view(),
                      output_train_nr(ci->train_nr_, ci->original_train_nr_),
                      ci->line_identifier_.view(),
                      ci->provider_ != nullptr
                          ? ci->provider_->full_name_.view()
                          : std::string_view{},
                      fc.clasz_};
}

std::vector<std::pair<service_info, unsigned>> get_service_infos(
    schedule const& sched, trip const* trp) {
  mcd::hash_map<service_info, unsigned> si_counts;
  for (auto const& section : motis::access::sections(trp)) {
    auto const& fc = section.fcon();
    for (auto ci = fc.con_info_; ci != nullptr; ci = ci->merged_with_) {
      auto const si = get_service_info(sched, fc, ci);
      ++si_counts[si];
    }
  }
  auto sis = utl::to_vec(si_counts, [](auto const& e) {
    return std::make_pair(e.first, e.second);
  });
  std::sort(begin(sis), end(sis),
            [](auto const& a, auto const& b) { return a.second > b.second; });
  return sis;
}

std::vector<std::pair<service_info, unsigned>> get_service_infos_for_leg(
    schedule const& sched, journey_leg const& leg) {
  auto const* trp = get_trip(sched, leg.trip_idx_);
  mcd::hash_map<service_info, unsigned> si_counts;
  auto enter_found = false;
  auto in_trip = false;
  for (auto const& section : motis::access::sections(trp)) {
    if (!in_trip) {
      if (section.from_station_id() == leg.enter_station_id_ &&
          get_schedule_time(sched, section.ev_key_from()) == leg.enter_time_) {
        in_trip = true;
        enter_found = true;
      } else {
        continue;
      }
    }
    auto const& fc = section.fcon();
    for (auto ci = fc.con_info_; ci != nullptr; ci = ci->merged_with_) {
      auto const si = get_service_info(sched, fc, ci);
      ++si_counts[si];
    }
    if (section.to_station_id() == leg.exit_station_id_ &&
        get_schedule_time(sched, section.ev_key_to()) == leg.exit_time_) {
      // in_trip = false;
      break;
    }
  }
  // it's possible that the enter and/or exit event are not found if this
  // is not a currently valid compact journey leg (e.g. a planned journey)
  // in that case, return all service infos for the trip
  if (!enter_found) {
    return get_service_infos(sched, trp);
  }
  auto sis = utl::to_vec(si_counts, [](auto const& e) {
    return std::pair{e.first, e.second};
  });
  std::sort(begin(sis), end(sis),
            [](auto const& a, auto const& b) { return a.second > b.second; });
  return sis;
}

}  // namespace motis::paxmon
