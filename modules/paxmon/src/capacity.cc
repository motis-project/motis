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

bool fill_trip_capacity(capacity_maps const& caps, trip_capacity& cap,
                        std::uint32_t train_nr) {
  auto const tid = get_cap_trip_id(cap.trp_->id_, train_nr);
  cap.trip_lookup_source_ = capacity_source::UNKNOWN;

  // try to match full trip id or primary trip id
  if (auto const lb = caps.trip_capacity_map_.lower_bound(tid);
      lb != end(caps.trip_capacity_map_)) {
    if (lb->first == tid) {
      cap.trip_lookup_capacity_.seats_ = lb->second;
      cap.trip_lookup_source_ = capacity_source::TRIP_EXACT;
    } else if (primary_trip_id_matches(lb->first, tid)) {
      cap.trip_lookup_capacity_.seats_ = lb->second;
      cap.trip_lookup_source_ = capacity_source::TRIP_PRIMARY;
    } else if (lb != begin(caps.trip_capacity_map_)) {
      if (auto const prev = std::prev(lb);
          primary_trip_id_matches(prev->first, tid)) {
        cap.trip_lookup_capacity_.seats_ = prev->second;
        cap.trip_lookup_source_ = capacity_source::TRIP_PRIMARY;
      }
    }
  }

  if (cap.has_trip_lookup_capacity() || caps.fuzzy_match_max_time_diff_ == 0) {
    return cap.has_trip_lookup_capacity();
  }

  // find the best possible match where train number matches
  auto const tid_train_nr_only = cap_trip_id{train_nr};
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
      cap.trip_lookup_capacity_.seats_ = it->second;
      cap.trip_lookup_source_ = stations_match(tid, cid)
                                    ? capacity_source::TRAIN_NR_AND_STATIONS
                                    : capacity_source::TRAIN_NR;
      best_station_matches = cur_station_matches;
      best_diff = cur_diff;
    }
  }

  return cap.has_trip_lookup_capacity();
}

bool fill_trip_capacity(schedule const& sched, capacity_maps const& caps,
                        trip_capacity& cap) {

  auto const trp_train_nr = cap.trp_->id_.primary_.get_train_nr();
  if (fill_trip_capacity(caps, cap, trp_train_nr)) {
    return true;
  }

  if (cap.con_info_->train_nr_ != trp_train_nr) {
    if (fill_trip_capacity(caps, cap, cap.con_info_->train_nr_)) {
      return true;
    }
  }

  auto const trp_line_nr = get_line_nr(cap.trp_->id_.secondary_.line_id_);
  if (trp_line_nr && *trp_line_nr != trp_train_nr) {
    if (fill_trip_capacity(caps, cap, *trp_line_nr)) {
      return true;
    }
  }

  auto const ci_line_nr = get_line_nr(cap.con_info_->line_identifier_);
  if (ci_line_nr && ci_line_nr != trp_line_nr && *ci_line_nr != trp_train_nr) {
    if (fill_trip_capacity(caps, cap, *ci_line_nr)) {
      return true;
    }
  }

  auto const& category = sched.categories_[cap.con_info_->family_]->name_;
  if (auto const it = caps.category_capacity_map_.find(category);
      it != end(caps.category_capacity_map_)) {
    cap.trip_lookup_capacity_.seats_ = it->second;
    cap.trip_lookup_source_ = capacity_source::CATEGORY;
    return true;
  } else if (auto const it = caps.category_capacity_map_.find(std::to_string(
                 static_cast<service_class_t>(cap.full_con_->clasz_)));
             it != end(caps.category_capacity_map_)) {
    cap.trip_lookup_capacity_.seats_ = it->second;
    cap.trip_lookup_source_ = capacity_source::CLASZ;
    return true;
  }

  return false;
}

trip_formation const* get_trip_formation(capacity_maps const& caps,
                                         trip const* trp) {
  if (auto const it = caps.trip_uuid_map_.find(trp->id_.primary_);
      it != end(caps.trip_uuid_map_)) {
    auto const trip_uuid = it->second;
    if (auto const it = caps.trip_formation_map_.find(trip_uuid);
        it != end(caps.trip_formation_map_)) {
      return &it->second;
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

std::pair<trip_formation const*, trip_formation_section const*>
get_trip_formation_section(schedule const& sched, capacity_maps const& caps,
                           trip const* trp, ev_key const& ev_key_from) {
  auto const* formation = get_trip_formation(caps, trp);
  if (formation == nullptr || formation->sections_.empty()) {
    return {nullptr, nullptr};
  }
  auto const schedule_departure = get_schedule_time(sched, ev_key_from);
  auto const& station_eva =
      sched.stations_[ev_key_from.get_station_idx()]->eva_nr_;
  auto const* best_section_match = &formation->sections_.front();
  auto station_found = false;
  for (auto const& sec : formation->sections_) {
    auto const station_match = sec.departure_eva_ == station_eva;
    if (station_match && sec.schedule_departure_time_ == schedule_departure) {
      return {formation, &sec};
    }
    if (!station_found) {
      station_found = station_match;
      if (station_found || sec.schedule_departure_time_ < schedule_departure) {
        best_section_match = &sec;
      }
    }
  }
  return {formation, best_section_match};
}

section_capacity get_section_capacity(schedule const& sched,
                                      capacity_maps const& caps,
                                      light_connection const& lc,
                                      ev_key const& ev_key_from,
                                      bool const detailed) {
  auto cap = section_capacity{.source_ = capacity_source::FORMATION_VEHICLES};
  auto processed_uics = mcd::hash_set<std::uint64_t>{};
  auto processed_vehicle_groups = mcd::hash_set<mcd::string>{};
  auto data_found = false;
  auto ci = lc.full_con_->con_info_;

  for (auto const& trp : *sched.merged_trips_.at(lc.trips_)) {
    auto const [tf, tf_sec] =
        get_trip_formation_section(sched, caps, trp, ev_key_from);
    auto& trp_cap =
        cap.trips_.emplace_back(trip_capacity{.trp_ = trp,
                                              .formation_ = tf,
                                              .formation_section_ = tf_sec,
                                              .full_con_ = lc.full_con_,
                                              .con_info_ = ci});
    ci = ci->merged_with_;
    if (tf_sec == nullptr) {
      continue;
    }

    auto trp_data_found = false;
    trp_cap.formation_source_ = capacity_source::FORMATION_VEHICLES;

    for (auto const& vg : tf_sec->vehicle_groups_) {
      auto const duplicate_group = processed_vehicle_groups.find(vg.name_) !=
                                   end(processed_vehicle_groups);
      if (duplicate_group && !detailed) {
        continue;
      }
      processed_vehicle_groups.insert(vg.name_);

      auto vg_cap = vehicle_group_capacity{
          .group_ = &vg, .trp_ = trp, .duplicate_group_ = duplicate_group};
      if (auto const it = caps.vehicle_group_capacity_map_.find(vg.name_);
          it != end(caps.vehicle_group_capacity_map_)) {
        vg_cap.capacity_ = it->second;
        vg_cap.source_ = capacity_source::FORMATION_VEHICLE_GROUPS;
      }

      auto vehicle_cap_sum = detailed_capacity{};
      auto all_uics_found = true;
      auto baureihe_used = false;
      auto gattung_used = false;
      for (auto const& vi : vg.vehicles_) {
        auto const duplicate_vehicle =
            vi.has_uic() && processed_uics.find(vi.uic_) != end(processed_uics);
        if (duplicate_vehicle && !detailed) {
          continue;
        } else if (vi.has_uic()) {
          processed_uics.insert(vi.uic_);
        }

        auto vehicle_cap = vehicle_capacity{
            .vehicle_ = &vi, .duplicate_vehicle_ = duplicate_vehicle};

        if (!duplicate_vehicle) {
          ++cap.num_vehicles_;
        }

        if (auto const it = caps.vehicle_capacity_map_.find(vi.uic_);
            it != end(caps.vehicle_capacity_map_) && vi.has_uic()) {
          vehicle_cap.capacity_ = it->second;
          vehicle_cap.source_ = capacity_source::FORMATION_VEHICLES;
          trp_data_found = true;
          if (!duplicate_vehicle) {
            ++cap.num_vehicles_uic_found_;
          }
        } else {
          all_uics_found = false;
          if (auto const it = caps.baureihe_capacity_map_.find(vi.baureihe_);
              it != end(caps.baureihe_capacity_map_)) {
            vehicle_cap.capacity_ = it->second;
            vehicle_cap.source_ = capacity_source::FORMATION_BAUREIHE;
            trp_data_found = true;
            baureihe_used = true;
            if (!duplicate_vehicle) {
              ++cap.num_vehicles_baureihe_used_;
            }
          } else if (auto const it =
                         caps.gattung_capacity_map_.find(vi.type_code_);
                     it != end(caps.gattung_capacity_map_)) {
            vehicle_cap.capacity_ = it->second;
            vehicle_cap.source_ = capacity_source::FORMATION_GATTUNG;
            trp_data_found = true;
            gattung_used = true;
            if (!duplicate_vehicle) {
              ++cap.num_vehicles_gattung_used_;
            }
          } else {
            if (!duplicate_vehicle) {
              ++cap.num_vehicles_no_info_;
            }
          }
        }
        if (!duplicate_vehicle) {
          vehicle_cap_sum += vehicle_cap.capacity_;
        }
        if (detailed) {
          vg_cap.vehicles_.emplace_back(vehicle_cap);
        }
      }

      if (!duplicate_group) {
        auto const update_capacity = [&](detailed_capacity const& c) {
          cap.capacity_ += c;
          trp_cap.formation_capacity_ += c;
        };
        auto const update_source = [&](capacity_source const src) {
          cap.source_ = get_worst_source(cap.source_, src);
          trp_cap.formation_source_ =
              get_worst_source(trp_cap.formation_source_, src);
        };

        if (!all_uics_found && vg_cap.capacity_.seats() != 0) {
          // if not all vehicle uics were found, but the vehicle group was
          // found, use the vehicle group info
          update_capacity(vg_cap.capacity_);
          update_source(capacity_source::FORMATION_VEHICLE_GROUPS);
          ++cap.num_vehicle_groups_used_;
          trp_data_found = true;
        } else {
          update_capacity(vehicle_cap_sum);
          if (vehicle_cap_sum.seats_ > 0) {
            vg_cap.capacity_ = vehicle_cap_sum;
            vg_cap.source_ = capacity_source::FORMATION_VEHICLES;
          }
          if (baureihe_used) {
            update_source(capacity_source::FORMATION_BAUREIHE);
            vg_cap.source_ = capacity_source::FORMATION_BAUREIHE;
          }
          if (gattung_used) {
            update_source(capacity_source::FORMATION_GATTUNG);
            vg_cap.source_ = capacity_source::FORMATION_GATTUNG;
          }
        }
      }

      if (detailed) {
        cap.vehicle_groups_.emplace_back(std::move(vg_cap));
      }
    }

    if (trp_data_found) {
      data_found = true;
    } else {
      trp_cap.formation_source_ = capacity_source::UNKNOWN;
    }
  }

  if (!data_found) {
    cap.source_ = capacity_source::UNKNOWN;
  }

  return cap;
}

std::optional<detailed_capacity> get_override_capacity(
    schedule const& sched, capacity_maps const& caps, trip const* trp,
    ev_key const& ev_key_from) {
  auto const tid = get_cap_trip_id(trp->id_);
  if (auto const it = caps.override_map_.find(tid);
      it != end(caps.override_map_)) {
    auto const schedule_departure = get_schedule_time(sched, ev_key_from);
    auto const departure_station = ev_key_from.get_station_idx();
    auto best_section_capacity = std::optional<detailed_capacity>{};
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

std::optional<detailed_capacity> get_override_capacity(
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

section_capacity get_capacity(schedule const& sched, light_connection const& lc,
                              ev_key const& ev_key_from,
                              ev_key const& /*ev_key_to*/,
                              capacity_maps const& caps, bool const detailed) {
  auto cap = get_section_capacity(sched, caps, lc, ev_key_from, detailed);

  if (!cap.has_capacity() || detailed) {
    auto capacity = detailed_capacity{};
    auto worst_source = capacity_source::TRIP_EXACT;
    auto some_unknown = false;

    for (auto& trp_cap : cap.trips_) {
      if (fill_trip_capacity(sched, caps, trp_cap)) {
        capacity += trp_cap.trip_lookup_capacity_;
        worst_source =
            get_worst_source(worst_source, trp_cap.trip_lookup_source_);
      } else {
        some_unknown = true;
      }
    }

    if (!cap.has_capacity() && !some_unknown) {
      cap.capacity_ = capacity;
      cap.source_ = worst_source;
    }
  }

  if (cap.has_capacity()) {
    cap.capacity_.seats_ = clamp_capacity(caps, cap.capacity_.seats());
  }

  auto const override_capacity =
      get_override_capacity(sched, caps, lc.trips_, ev_key_from);
  if (override_capacity.has_value()) {
    cap.original_capacity_ = cap.capacity_;
    cap.original_source_ = cap.source_;

    cap.capacity_ = *override_capacity;
    cap.source_ = capacity_source::OVERRIDE;
  }

  return cap;
}

}  // namespace motis::paxmon
