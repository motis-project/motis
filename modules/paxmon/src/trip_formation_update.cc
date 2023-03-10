#include "motis/paxmon/trip_formation_update.h"

#include <algorithm>
#include <string_view>

#include "motis/rt/util.h"
#include "motis/vector.h"

#include "motis/paxmon/update_capacity.h"

using namespace motis::rt;
using namespace motis::ris;

namespace motis::paxmon {

inline mcd::string fbs_to_mcd_str(flatbuffers::String const* s) {
  return mcd::string{std::string_view{s->c_str(), s->size()}};
}

mcd::vector<vehicle_info> get_section_vehicles(
    TripFormationSection const* sec) {
  auto vehicles = mcd::vector<vehicle_info>{};
  for (auto const& vg : *sec->vehicle_groups()) {
    for (auto const& vi : *vg->vehicles()) {
      auto const uic = vi->uic();
      if (std::find_if(begin(vehicles), end(vehicles), [&](auto const& v) {
            return v.uic_ == uic;
          }) == end(vehicles)) {
        vehicles.emplace_back(vehicle_info{uic, fbs_to_mcd_str(vi->baureihe()),
                                           fbs_to_mcd_str(vi->type_code()),
                                           fbs_to_mcd_str(vi->order())});
      }
    }
  }
  return vehicles;
}

void update_trip_formation(schedule const& sched, universe& uv,
                           motis::ris::TripFormationMessage const* tfm) {
  auto const trip_uuid = parse_uuid(view(tfm->trip_id()->uuid()));
  primary_trip_id ptid;
  if (get_primary_trip_id(sched, tfm->trip_id(), ptid)) {
    uv.capacity_maps_.trip_uuid_map_[ptid] = trip_uuid;
  }

  auto& formation = uv.capacity_maps_.trip_formation_map_[trip_uuid];
  formation.sections_ =
      mcd::to_vec(*tfm->sections(), [&](TripFormationSection const* sec) {
        return trip_formation_section{
            view(sec->departure_station()->eva()),
            unix_to_motistime(sched.schedule_begin_,
                              sec->schedule_departure_time()),
            get_section_vehicles(sec)};
      });

  if (auto const it = sched.uuid_to_trip_.find(trip_uuid);
      it != end(sched.uuid_to_trip_)) {
    update_trip_capacity(uv, sched, it->second);
  }
}

}  // namespace motis::paxmon
