#include "motis/paxmon/api/get_trip_capacity.h"

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "boost/uuid/uuid_io.hpp"

#include "motis/hash_set.h"

#include "motis/core/access/realtime_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/capacity_internal.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/load_info.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/service_info.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace flatbuffers;

namespace motis::paxmon::api {

mcd::hash_set<trip const*> collect_merged_trips(
    schedule const& sched, Vector<Offset<TripId>> const* req_trips) {
  auto trips = mcd::hash_set<trip const*>{};

  for (auto const& req_trip : *req_trips) {
    auto const* trp = from_fbs(sched, req_trip);
    trips.insert(trp);
    for (auto const& sec : access::sections{trp}) {
      for (auto const& merged_trp :
           *sched.merged_trips_.at(sec.lcon().trips_)) {
        trips.insert(merged_trp);
      }
    }
  }

  return trips;
}

Offset<PaxMonCapacityData> to_fbs_capacity_data(FlatBufferBuilder& fbb,
                                                detailed_capacity const& vc) {
  return CreatePaxMonCapacityData(fbb, vc.limit(), vc.seats_, vc.seats_1st_,
                                  vc.seats_2nd_, vc.standing_, vc.total_limit_);
}

Offset<Station> to_optional_fbs_station(FlatBufferBuilder& fbb,
                                        schedule const& sched,
                                        mcd::string const& eva) {
  if (auto const* st = find_station(sched, eva); st != nullptr) {
    return to_fbs(fbb, *st);
  } else {
    auto const pos = Position{0, 0};
    return CreateStation(fbb, fbb.CreateString(eva), fbb.CreateSharedString(""),
                         &pos);
  }
}

msg_ptr get_trip_capacity(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonGetTripCapacityRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;
  auto const& caps = uv.capacity_maps_;

  auto const trips = collect_merged_trips(sched, req->trips());

  message_creator mc;

  auto const section_to_fbs = [&](access::trip_section const& sec) {
    auto const cap = get_capacity(sched, sec.lcon(), sec.ev_key_from(),
                                  sec.ev_key_to(), caps, true);

    auto const merged_trip_infos =
        utl::to_vec(cap.trips_, [&](trip_capacity const& trp_cap) {
          auto vehicle_groups = std::vector<Offset<PaxMonVehicleGroupInfo>>{};
          for (auto const& vg : cap.vehicle_groups_) {
            if (vg.trp_ != trp_cap.trp_) {
              continue;
            }

            auto fbs_vg_cap = std::vector<Offset<PaxMonCapacityData>>{};
            if (vg.capacity_.seats() != 0) {
              fbs_vg_cap.emplace_back(to_fbs_capacity_data(mc, vg.capacity_));
            }

            auto vehicles =
                utl::to_vec(vg.vehicles_, [&](vehicle_capacity const& vc) {
                  auto const uic_found =
                      vc.source_ == capacity_source::FORMATION_VEHICLES;
                  auto const guessed =
                      !uic_found && vc.source_ != capacity_source::UNKNOWN;
                  return CreatePaxMonVehicleCapacityInfo(
                      mc, vc.vehicle_->uic_, uic_found, guessed,
                      mc.CreateSharedString(vc.vehicle_->baureihe_.str()),
                      mc.CreateSharedString(vc.vehicle_->type_code_.str()),
                      mc.CreateSharedString(vc.vehicle_->order_.str()),
                      to_fbs_capacity_data(mc, vc.capacity_),
                      to_fbs_capacity_source(vc.source_));
                });

            vehicle_groups.emplace_back(CreatePaxMonVehicleGroupInfo(
                mc, mc.CreateSharedString(vg.group_->name_.str()),
                to_optional_fbs_station(mc, sched, vg.group_->start_eva_),
                to_optional_fbs_station(mc, sched, vg.group_->destination_eva_),
                mc.CreateSharedString(
                    boost::uuids::to_string(vg.group_->trip_uuid_)),
                to_fbs(mc, vg.group_->primary_trip_id_),
                mc.CreateVector(fbs_vg_cap), mc.CreateVector(vehicles)));
          }

          auto override = std::vector<Offset<PaxMonCapacityData>>{};
          if (auto const override_cap = get_override_capacity(
                  sched, caps, trp_cap.trp_, sec.ev_key_from());
              override_cap) {
            override.emplace_back(to_fbs_capacity_data(mc, *override_cap));
          }

          return CreatePaxMonMergedTripCapacityInfo(
              mc, to_fbs(sched, mc, trp_cap.trp_),
              to_fbs(mc, get_service_info(sched, *trp_cap.full_con_,
                                          trp_cap.con_info_)),
              to_fbs_capacity_data(mc, trp_cap.trip_lookup_capacity_),
              to_fbs_capacity_source(trp_cap.trip_lookup_source_),
              to_fbs_capacity_data(mc, trp_cap.formation_capacity_),
              to_fbs_capacity_source(trp_cap.formation_source_),
              trp_cap.has_formation(), mc.CreateVector(vehicle_groups),
              mc.CreateVector(override));
        });

    return CreatePaxMonSectionCapacityInfo(
        mc, to_fbs(mc, sec.from_station(sched)),
        to_fbs(mc, sec.to_station(sched)),
        motis_to_unixtime(sched, get_schedule_time(sched, sec.ev_key_from())),
        motis_to_unixtime(sched, sec.lcon().d_time_),
        motis_to_unixtime(sched, get_schedule_time(sched, sec.ev_key_to())),
        motis_to_unixtime(sched, sec.lcon().a_time_),
        cap.has_capacity() ? PaxMonCapacityType_Known
                           : PaxMonCapacityType_Unknown,
        to_fbs_capacity_data(mc, cap.capacity_),
        to_fbs_capacity_source(cap.source_),
        mc.CreateVector(merged_trip_infos));
  };

  auto const trip_to_fbs = [&](trip const* trp) {
    auto const tdi = uv.trip_data_.find_index(trp->trip_idx_);
    return CreatePaxMonTripCapacityInfo(
        mc, to_fbs_trip_service_info(mc, sched, trp),
        tdi != INVALID_TRIP_DATA_INDEX
            ? to_fbs(mc, uv.trip_data_.capacity_status(tdi))
            : to_fbs(mc, trip_capacity_status{}),
        mc.CreateVector(utl::to_vec(access::sections{trp}, section_to_fbs)));
  };

  mc.create_and_finish(
      MsgContent_PaxMonGetTripCapacityResponse,
      CreatePaxMonGetTripCapacityResponse(
          mc, mc.CreateVector(utl::to_vec(trips, trip_to_fbs)),
          caps.min_capacity_, caps.fuzzy_match_max_time_diff_,
          caps.trip_capacity_map_.size(), caps.category_capacity_map_.size(),
          caps.vehicle_capacity_map_.size(), caps.trip_formation_map_.size(),
          caps.override_map_.size(), caps.baureihe_capacity_map_.size(),
          caps.gattung_capacity_map_.size(),
          caps.vehicle_group_capacity_map_.size())
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
