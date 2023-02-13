#include "motis/paxmon/capacity.h"

#include <charconv>
#include <cstdint>
#include <ctime>
#include <algorithm>
#include <functional>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

#include "motis/core/common/logging.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/uuids.h"

#include "motis/paxmon/util/get_station_idx.h"

using namespace motis::logging;
using namespace motis::paxmon::util;

namespace motis::paxmon {

std::optional<std::pair<std::uint16_t, capacity_source>> get_trip_capacity(
    trip_capacity_map_t const& trip_map, trip const* trp,
    std::uint32_t train_nr) {
  auto const tid = cap_trip_id{train_nr, trp->id_.primary_.get_station_id(),
                               trp->id_.secondary_.target_station_id_,
                               trp->id_.primary_.get_time(),
                               trp->id_.secondary_.target_time_};
  if (auto const lb = trip_map.lower_bound(tid); lb != end(trip_map)) {
    if (lb->first == tid) {
      return {{lb->second, capacity_source::TRIP_EXACT}};
    } else if (lb->first.train_nr_ == train_nr) {
      return {{lb->second, capacity_source::TRAIN_NR}};
    } else if (lb != begin(trip_map)) {
      if (auto const prev = std::prev(lb);
          prev != end(trip_map) && prev->first.train_nr_ == train_nr) {
        return {{prev->second, capacity_source::TRAIN_NR}};
      }
    }
  }
  return {};
}

std::optional<std::uint32_t> get_line_nr(mcd::string const& line_id) {
  std::uint32_t line_nr = 0;
  auto const result =
      std::from_chars(line_id.data(), line_id.data() + line_id.size(), line_nr);
  if (result.ec == std::errc{} &&
      result.ptr == line_id.data() + line_id.size()) {
    return {line_nr};
  } else {
    return {};
  }
}

std::optional<std::pair<std::uint16_t, capacity_source>> get_trip_capacity(
    schedule const& sched, trip_capacity_map_t const& trip_map,
    category_capacity_map_t const& category_map, trip const* trp,
    connection_info const* ci, service_class const clasz) {

  auto const trp_train_nr = trp->id_.primary_.get_train_nr();
  if (auto const trip_capacity = get_trip_capacity(trip_map, trp, trp_train_nr);
      trip_capacity) {
    return trip_capacity;
  }

  if (ci->train_nr_ != trp_train_nr) {
    if (auto const trip_capacity =
            get_trip_capacity(trip_map, trp, ci->train_nr_);
        trip_capacity) {
      return trip_capacity;
    }
  }

  auto const trp_line_nr = get_line_nr(trp->id_.secondary_.line_id_);
  if (trp_line_nr && *trp_line_nr != trp_train_nr) {
    if (auto const trip_capacity =
            get_trip_capacity(trip_map, trp, *trp_line_nr);
        trip_capacity) {
      return trip_capacity;
    }
  }

  auto const ci_line_nr = get_line_nr(ci->line_identifier_);
  if (ci_line_nr && ci_line_nr != trp_line_nr && *ci_line_nr != trp_train_nr) {
    if (auto const trip_capacity =
            get_trip_capacity(trip_map, trp, *ci_line_nr);
        trip_capacity) {
      return trip_capacity;
    }
  }

  auto const& category = sched.categories_[ci->family_]->name_;
  if (auto const it = category_map.find(category); it != end(category_map)) {
    return {{it->second, capacity_source::CATEGORY}};
  } else if (auto const it = category_map.find(
                 std::to_string(static_cast<service_class_t>(clasz)));
             it != end(category_map)) {
    return {{it->second, capacity_source::CLASZ}};
  }

  return {};
}

inline capacity_source get_worst_source(capacity_source const a,
                                        capacity_source const b) {
  return static_cast<capacity_source>(
      std::max(static_cast<std::underlying_type_t<capacity_source>>(a),
               static_cast<std::underlying_type_t<capacity_source>>(b)));
}

std::pair<std::uint16_t, capacity_source> get_capacity(
    schedule const& sched, light_connection const& lc,
    ev_key const& /*ev_key_from*/, ev_key const& /*ev_key_to*/,
    capacity_maps const& caps) {
  std::uint16_t capacity = 0;
  auto worst_source = capacity_source::TRIP_EXACT;

  auto ci = lc.full_con_->con_info_;
  for (auto const& trp : *sched.merged_trips_.at(lc.trips_)) {
    utl::verify(ci != nullptr, "get_capacity: missing connection_info");

    auto const trip_capacity = get_trip_capacity(sched, caps.trip_capacity_map_,
                                                 caps.category_capacity_map_,
                                                 trp, ci, lc.full_con_->clasz_);
    if (trip_capacity.has_value()) {
      capacity += trip_capacity->first;
      worst_source = get_worst_source(worst_source, trip_capacity->second);
    } else {
      return {UNKNOWN_CAPACITY, capacity_source::SPECIAL};
    }

    ci = ci->merged_with_;
  }

  return {capacity, worst_source};
}

}  // namespace motis::paxmon
