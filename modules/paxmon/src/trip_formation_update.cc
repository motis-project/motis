#include "motis/paxmon/trip_formation_update.h"

#include <algorithm>
#include <string_view>

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
  auto sec = trip_formation_section{
      .departure_eva_ = view(tfs->departure_station()->eva()),
      .schedule_departure_time_ = unix_to_motistime(
          sched.schedule_begin_, tfs->schedule_departure_time()),
      .vehicles_ = {},
      .vehicle_groups_ = {}};
  for (auto const& vg : *tfs->vehicle_groups()) {
    auto const vg_idx = static_cast<std::uint8_t>(sec.vehicle_groups_.size());
    sec.vehicle_groups_.emplace_back(vehicle_group{
        .name_ = fbs_to_mcd_str(vg->name()),
        .start_eva_ = fbs_to_mcd_str(vg->start_station()->eva()),
        .destination_eva_ = fbs_to_mcd_str(vg->destination_station()->eva()),
        .trip_uuid_ = parse_uuid(view(vg->trip_id()->uuid())),
        .primary_trip_id_ = to_extern_trip(vg->trip_id()->id())});
    for (auto const& vi : *vg->vehicles()) {
      auto const uic = vi->uic();
      if (auto it = std::find_if(begin(sec.vehicles_), end(sec.vehicles_),
                                 [&](auto const& v) { return v.uic_ == uic; });
          it == end(sec.vehicles_)) {
        sec.vehicles_.emplace_back(
            vehicle_info{.uic_ = uic,
                         .baureihe_ = fbs_to_mcd_str(vi->baureihe()),
                         .type_code_ = fbs_to_mcd_str(vi->type_code()),
                         .order_ = fbs_to_mcd_str(vi->order()),
                         .vehicle_groups_ = {vg_idx}});
      } else if (std::find(begin(it->vehicle_groups_), end(it->vehicle_groups_),
                           vg_idx) == end(it->vehicle_groups_)) {
        it->vehicle_groups_.emplace_back(vg_idx);
      }
    }
  }
  return sec;
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
        return to_trip_formation_section(sched, sec);
      });

  if (auto const it = sched.uuid_to_trip_.find(trip_uuid);
      it != end(sched.uuid_to_trip_)) {
    update_trip_capacity(uv, sched, it->second);
  }
}

}  // namespace motis::paxmon
