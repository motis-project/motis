#include "motis/paxmon/capacity.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

#include "motis/hash_set.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/station_access.h"

#include "motis/core/debug/trip.h"

#include "motis/paxmon/capacity_internal.h"
#include "motis/paxmon/util/get_station_idx.h"

using namespace motis::paxmon::util;

namespace motis::paxmon {

struct debug_cap_trip_id {
  friend std::ostream& operator<<(std::ostream& out,
                                  debug_cap_trip_id const& t) {
    auto const& sched = t.sched_;
    auto const& id = t.id_;
    return out << "{train_nr=" << id.train_nr_ << ", from_station="
               << debug::station{sched, id.from_station_idx_}
               << ", to_station=" << debug::station{sched, id.to_station_idx_}
               << ", departure=" << format_time(id.departure_)
               << ", arrival=" << format_time(id.arrival_) << "}";
  }

  schedule const& sched_;
  cap_trip_id id_;
};

bool primary_trip_id_matches(cap_trip_id const& a, cap_trip_id const& b) {
  return a.train_nr_ == b.train_nr_ &&
         a.from_station_idx_ == b.from_station_idx_ &&
         a.departure_ == b.departure_;
}

bool stations_match(cap_trip_id const& a, cap_trip_id const& b) {
  return a.from_station_idx_ == b.from_station_idx_ &&
         a.to_station_idx_ == b.to_station_idx_;
}

std::optional<std::pair<std::uint16_t, capacity_source>> get_trip_capacity(
    schedule const& /*sched*/, capacity_maps const& caps, trip const* trp,
    std::uint32_t train_nr) {
  auto const tid = get_cap_trip_id(trp->id_, train_nr);
  // try to match full trip id or primary trip id
  if (auto const lb = caps.trip_capacity_map_.lower_bound(tid);
      lb != end(caps.trip_capacity_map_)) {
    if (lb->first == tid) {
      return {{lb->second, capacity_source::TRIP_EXACT}};
    }
    if (primary_trip_id_matches(lb->first, tid)) {
      return {{lb->second, capacity_source::TRIP_PRIMARY}};
    } else if (lb != begin(caps.trip_capacity_map_)) {
      if (auto const prev = std::prev(lb);
          primary_trip_id_matches(prev->first, tid)) {
        return {{prev->second, capacity_source::TRIP_PRIMARY}};
      }
    }
  }

  if (caps.fuzzy_match_max_time_diff_ == 0) {
    return {};
  }

  // find the best possible match where train number matches
  auto const tid_train_nr_only = cap_trip_id{train_nr};
  auto best_result = std::optional<std::pair<std::uint16_t, capacity_source>>{};
  auto best_station_matches = 0;
  auto best_diff = std::numeric_limits<int>::max();
  for (auto it = caps.trip_capacity_map_.lower_bound(tid_train_nr_only);
       it != end(caps.trip_capacity_map_) && it->first.train_nr_ == train_nr;
       it = std::next(it)) {
    auto const& cid = it->first;
    auto const cur_dep_diff = std::abs(static_cast<int>(cid.departure_) -
                                       static_cast<int>(tid.departure_));
    auto const cur_arr_diff = std::abs(static_cast<int>(cid.arrival_) -
                                       static_cast<int>(tid.arrival_));

    if (cur_dep_diff > caps.fuzzy_match_max_time_diff_ ||
        cur_arr_diff > caps.fuzzy_match_max_time_diff_) {
      continue;
    }

    auto const cur_from_station_match =
        cid.from_station_idx_ == tid.from_station_idx_;
    auto const cur_to_station_match =
        cid.to_station_idx_ == tid.to_station_idx_;
    auto const cur_station_matches =
        (cur_from_station_match ? 1 : 0) + (cur_to_station_match ? 1 : 0);
    auto const cur_diff = cur_dep_diff + cur_arr_diff;

    auto const better =
        (cur_station_matches > best_station_matches) ||
        (cur_station_matches == best_station_matches && cur_diff < best_diff);

    if (better) {
      best_result = {{it->second, stations_match(tid, cid)
                                      ? capacity_source::TRAIN_NR_AND_STATIONS
                                      : capacity_source::TRAIN_NR}};
      best_station_matches = cur_station_matches;
      best_diff = cur_diff;
    }
  }

  return best_result;
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
    schedule const& sched, capacity_maps const& caps, trip const* trp,
    connection_info const* ci, service_class const clasz) {

  auto const trp_train_nr = trp->id_.primary_.get_train_nr();
  if (auto const trip_capacity =
          get_trip_capacity(sched, caps, trp, trp_train_nr);
      trip_capacity) {
    return trip_capacity;
  }

  if (ci->train_nr_ != trp_train_nr) {
    if (auto const trip_capacity =
            get_trip_capacity(sched, caps, trp, ci->train_nr_);
        trip_capacity) {
      return trip_capacity;
    }
  }

  auto const trp_line_nr = get_line_nr(trp->id_.secondary_.line_id_);
  if (trp_line_nr && *trp_line_nr != trp_train_nr) {
    if (auto const trip_capacity =
            get_trip_capacity(sched, caps, trp, *trp_line_nr);
        trip_capacity) {
      return trip_capacity;
    }
  }

  auto const ci_line_nr = get_line_nr(ci->line_identifier_);
  if (ci_line_nr && ci_line_nr != trp_line_nr && *ci_line_nr != trp_train_nr) {
    if (auto const trip_capacity =
            get_trip_capacity(sched, caps, trp, *ci_line_nr);
        trip_capacity) {
      return trip_capacity;
    }
  }

  auto const& category = sched.categories_[ci->family_]->name_;
  if (auto const it = caps.category_capacity_map_.find(category);
      it != end(caps.category_capacity_map_)) {
    return {{it->second, capacity_source::CATEGORY}};
  } else if (auto const it = caps.category_capacity_map_.find(
                 std::to_string(static_cast<service_class_t>(clasz)));
             it != end(caps.category_capacity_map_)) {
    return {{it->second, capacity_source::CLASZ}};
  }

  return {};
}

trip_formation const* get_trip_formation(capacity_maps const& caps,
                                         trip const* trp) {
  auto trip_uuid = trp->uuid_;
  if (trip_uuid.is_nil()) {
    if (auto const it = caps.trip_uuid_map_.find(trp->id_.primary_);
        it != end(caps.trip_uuid_map_)) {
      trip_uuid = it->second;
    } else {
      return nullptr;
    }
  }
  if (auto const it = caps.trip_formation_map_.find(trip_uuid);
      it != end(caps.trip_formation_map_)) {
    return &it->second;
  } else {
    return nullptr;
  }
}

trip_formation_section const* get_trip_formation_section(
    schedule const& sched, capacity_maps const& caps, trip const* trp,
    ev_key const& ev_key_from) {
  auto const* formation = get_trip_formation(caps, trp);
  if (formation == nullptr || formation->sections_.empty()) {
    return nullptr;
  }
  auto const schedule_departure = get_schedule_time(sched, ev_key_from);
  auto const& station_eva =
      sched.stations_[ev_key_from.get_station_idx()]->eva_nr_;
  auto const* best_section_match = &formation->sections_.front();
  auto station_found = false;
  for (auto const& sec : formation->sections_) {
    auto const station_match = sec.departure_eva_ == station_eva;
    if (station_match && sec.schedule_departure_time_ == schedule_departure) {
      return &sec;
    }
    if (!station_found) {
      station_found = station_match;
      if (station_found || sec.schedule_departure_time_ < schedule_departure) {
        best_section_match = &sec;
      }
    }
  }
  return best_section_match;
}

std::optional<vehicle_capacity> get_section_capacity(
    schedule const& sched, capacity_maps const& caps,
    std::uint32_t const merged_trips_idx, ev_key const& ev_key_from) {
  auto uics = mcd::hash_set<std::uint64_t>{};
  for (auto const& trp : *sched.merged_trips_.at(merged_trips_idx)) {
    auto const* tf_sec =
        get_trip_formation_section(sched, caps, trp, ev_key_from);
    if (tf_sec != nullptr) {
      for (auto const& vi : tf_sec->vehicles_) {
        uics.insert(vi.uic_);
      }
    }
  }

  auto cap = vehicle_capacity{};
  for (auto const& uic : uics) {
    if (auto const it = caps.vehicle_capacity_map_.find(uic);
        it != end(caps.vehicle_capacity_map_)) {
      cap += it->second;
    }
  }
  if (cap.seats() > 0) {
    return cap;
  } else {
    return {};
  }
}

std::optional<vehicle_capacity> get_override_capacity(
    schedule const& sched, capacity_maps const& caps, trip const* trp,
    ev_key const& ev_key_from) {
  auto const tid = get_cap_trip_id(trp->id_);
  if (auto const it = caps.override_map_.find(tid);
      it != end(caps.override_map_)) {
    auto const schedule_departure = get_schedule_time(sched, ev_key_from);
    auto const departure_station = ev_key_from.get_station_idx();
    auto best_section_capacity = std::optional<vehicle_capacity>{};
    auto station_found = false;
    for (auto const& sec : it->second) {
      auto const station_match =
          sec.departure_station_idx_ == departure_station;
      if (station_match && sec.schedule_departure_time_ == schedule_departure) {
        return sec.total_capacity_;
      }
      if (!station_found) {
        station_found = station_match;
        if (station_found ||
            sec.schedule_departure_time_ < schedule_departure) {
          best_section_capacity = sec.total_capacity_;
        }
      }
    }
    return best_section_capacity;
  }
  return {};
}

std::optional<vehicle_capacity> get_override_capacity(
    schedule const& sched, capacity_maps const& caps,
    std::uint32_t const merged_trips_idx, ev_key const& ev_key_from) {
  for (auto const& trp : *sched.merged_trips_.at(merged_trips_idx)) {
    auto const result = get_override_capacity(sched, caps, trp, ev_key_from);
    if (result) {
      return result;
    }
  }
  return {};
}

std::pair<std::uint16_t, capacity_source> get_capacity(
    schedule const& sched, light_connection const& lc,
    ev_key const& ev_key_from, ev_key const& /*ev_key_to*/,
    capacity_maps const& caps) {
  std::uint16_t capacity = 0;
  auto worst_source = capacity_source::TRIP_EXACT;
  auto some_unknown = false;

  auto const override_capacity =
      get_override_capacity(sched, caps, lc.trips_, ev_key_from);
  if (override_capacity.has_value()) {
    return {override_capacity->seats(), capacity_source::TRIP_EXACT};
  }

  auto const section_capacity =
      get_section_capacity(sched, caps, lc.trips_, ev_key_from);
  if (section_capacity.has_value()) {
    return {clamp_capacity(caps, section_capacity->seats()),
            capacity_source::TRIP_EXACT};
  }

  auto ci = lc.full_con_->con_info_;
  for (auto const& trp : *sched.merged_trips_.at(lc.trips_)) {
    utl::verify(ci != nullptr, "get_capacity: missing connection_info");

    auto const trip_capacity =
        get_trip_capacity(sched, caps, trp, ci, lc.full_con_->clasz_);
    if (trip_capacity.has_value()) {
      capacity += trip_capacity->first;
      worst_source = get_worst_source(worst_source, trip_capacity->second);
    } else {
      some_unknown = true;
    }

    ci = ci->merged_with_;
  }

  if (some_unknown) {
    return {UNKNOWN_CAPACITY, capacity_source::SPECIAL};
  } else {
    return {clamp_capacity(caps, capacity), worst_source};
  }
}

}  // namespace motis::paxmon
