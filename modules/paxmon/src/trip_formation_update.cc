#include "motis/paxmon/trip_formation_update.h"

#include <algorithm>
#include <iostream>
#include <string_view>

#include "boost/uuid/uuid_io.hpp"

#include "motis/core/common/date_time_util.h"
#include "motis/core/debug/trip.h"

#include "motis/rt/util.h"
#include "motis/vector.h"

#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/update_capacity.h"

using namespace motis::rt;
using namespace motis::ris;

namespace motis::paxmon {

inline mcd::string fbs_to_mcd_str(flatbuffers::String const* s) {
  return mcd::string{std::string_view{s->c_str(), s->size()}};
}

trip_formation_section to_trip_formation_section(
    schedule const& sched, motis::ris::TripFormationSection const* tfs) {
  return trip_formation_section{
      .departure_eva_ = view(tfs->departure_station()->eva()),
      .schedule_departure_time_ = unix_to_motistime(
          sched.schedule_begin_, tfs->schedule_departure_time()),
      .vehicle_groups_ =
          mcd::to_vec(*tfs->vehicle_groups(), [](VehicleGroup const* fbs_vg) {
            return vehicle_group{
                .name_ = fbs_to_mcd_str(fbs_vg->name()),
                .start_eva_ = fbs_to_mcd_str(fbs_vg->start_station()->eva()),
                .destination_eva_ =
                    fbs_to_mcd_str(fbs_vg->destination_station()->eva()),
                .trip_uuid_ = parse_uuid(view(fbs_vg->trip_id()->uuid())),
                .primary_trip_id_ = to_extern_trip(fbs_vg->trip_id()->id()),
                .vehicles_ =
                    mcd::to_vec(*fbs_vg->vehicles(), [](VehicleInfo const* vi) {
                      return vehicle_info{
                          .uic_ = vi->uic(),
                          .baureihe_ = fbs_to_mcd_str(vi->baureihe()),
                          .type_code_ = fbs_to_mcd_str(vi->type_code()),
                          .order_ = fbs_to_mcd_str(vi->order())};
                    })};
          })};
}

trip* find_trip_by_primary_trip_id(schedule const& sched,
                                   primary_trip_id const& ptid,
                                   boost::uuids::uuid const& trip_uuid) {
  trip* result = nullptr;
  auto matching_trips = 0;
  for (auto it =
           std::lower_bound(begin(sched.trips_), end(sched.trips_),
                            std::make_pair(ptid, static_cast<trip*>(nullptr)));
       it != end(sched.trips_) && it->first == ptid; ++it) {
    result = it->second;
    ++matching_trips;
  }
  if (matching_trips > 1) {
    std::cout << "[UTF-06] found " << matching_trips
              << " matching trips by primary id: (formation trip uuid: "
              << trip_uuid << ")" << std::endl;
    for (auto it = std::lower_bound(
             begin(sched.trips_), end(sched.trips_),
             std::make_pair(ptid, static_cast<trip*>(nullptr)));
         it != end(sched.trips_) && it->first == ptid; ++it) {
      std::cout << "  " << debug::trip{sched, it->second} << std::endl;
    }
  }
  // only return trip if there is an unambiguous match
  return matching_trips == 1 ? result : nullptr;
}

void update_trip_formation(schedule const& sched, universe& uv,
                           motis::ris::TripFormationMessage const* tfm) {
  auto const trip_uuid = parse_uuid(view(tfm->trip_id()->uuid()));
  primary_trip_id ptid;
  auto const has_ptid = get_primary_trip_id(sched, tfm->trip_id(), ptid);
  if (has_ptid) {
    if (auto const it = uv.capacity_maps_.trip_uuid_map_.find(ptid);
        it != end(uv.capacity_maps_.trip_uuid_map_)) {
      if (it->second != trip_uuid) {
        std::cout << "[UTF-01] trip uuid CHANGED: " << it->second << " -> "
                  << trip_uuid << "\n  ptid: train_nr=" << ptid.get_train_nr()
                  << ", station="
                  << sched.stations_[ptid.get_station_id()]->name_
                  << ", time=" << format_time(ptid.get_time()) << std::endl;
      }
      if (auto const tf_it =
              uv.capacity_maps_.trip_formation_map_.find(trip_uuid);
          tf_it == end(uv.capacity_maps_.trip_formation_map_)) {
        std::cout << "[UTF-02] trip primary id found, but uuid not found: uuid="
                  << trip_uuid << ", train_nr=" << ptid.get_train_nr()
                  << ", station="
                  << sched.stations_[ptid.get_station_id()]->name_
                  << ", time=" << format_time(ptid.get_time()) << std::endl;
      }
    } else {
      if (auto const tf_it =
              uv.capacity_maps_.trip_formation_map_.find(trip_uuid);
          tf_it != end(uv.capacity_maps_.trip_formation_map_)) {
        std::cout << "[UTF-03] trip primary id not found, but uuid found: uuid="
                  << trip_uuid << ", train_nr=" << ptid.get_train_nr()
                  << ", station="
                  << sched.stations_[ptid.get_station_id()]->name_
                  << ", time=" << format_time(ptid.get_time()) << std::endl;
        if (auto const prev_ptid_it =
                uv.capacity_maps_.uuid_trip_map_.find(trip_uuid);
            prev_ptid_it != end(uv.capacity_maps_.uuid_trip_map_)) {
          auto const& prev_ptid = prev_ptid_it->second;
          std::cout << "  previous trip id: train_nr="
                    << prev_ptid.get_train_nr() << ", station="
                    << sched.stations_[prev_ptid.get_station_id()]->name_
                    << ", time=" << format_time(prev_ptid.get_time())
                    << std::endl;
        } else {
          std::cout << "  previous trip id not found" << std::endl;
        }
      }
    }
    uv.capacity_maps_.trip_uuid_map_[ptid] = trip_uuid;
    if (uv.capacity_maps_.uuid_trip_map_.find(trip_uuid) ==
        end(uv.capacity_maps_.uuid_trip_map_)) {
      uv.capacity_maps_.uuid_trip_map_[trip_uuid] = ptid;
    }
  } else {
    auto const& tid = tfm->trip_id()->id();
    std::cout << "[UTF-04] station from trip id not found: {station_id="
              << tid->station_id()->str() << ", train_nr=" << tid->train_nr()
              << ", time=" << tid->time() << " ("
              << format_unix_time(tid->time())
              << ")}, uuid=" << tfm->trip_id()->uuid()->str() << ", "
              << tfm->sections()->size() << " sections" << std::endl;
  }

  auto& formation = uv.capacity_maps_.trip_formation_map_[trip_uuid];
  formation.ptid_ = ptid;
  formation.uuid_ = trip_uuid;
  formation.category_ = fbs_to_mcd_str(tfm->trip_id()->category());
  formation.sections_ =
      mcd::to_vec(*tfm->sections(), [&](TripFormationSection const* sec) {
        return to_trip_formation_section(sched, sec);
      });

  if (has_ptid) {
    if (auto* trp = find_trip_by_primary_trip_id(sched, ptid, trip_uuid);
        trp != nullptr) {
      update_trip_capacity(uv, sched, trp);
      return;
    }
  }

  // if the primary trip id in the formation message has changed
  // (but uuid stayed the same), we won't find the trip using the new primary
  // trip id lookup (because motis trip ids never change)
  // maybe store trip <-> formation uuid mapping instead

  if (auto const it = uv.capacity_maps_.uuid_trip_map_.find(trip_uuid);
      it != end(uv.capacity_maps_.uuid_trip_map_)) {
    if (auto* trp = find_trip_by_primary_trip_id(sched, it->second, trip_uuid);
        trp != nullptr) {
      std::cout << "[UTF-08] found trip by previous primary id: uuid="
                << trip_uuid << std::endl;
      update_trip_capacity(uv, sched, trp);
    }
  }
}

}  // namespace motis::paxmon
