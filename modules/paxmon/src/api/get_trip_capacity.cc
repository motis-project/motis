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
                                                vehicle_capacity const& vc) {
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
    auto merged_trip_infos =
        std::vector<Offset<PaxMonMergedTripCapacityInfo>>{};
    auto const& lc = sec.lcon();
    auto ci = lc.full_con_->con_info_;
    for (auto const& trp : *sched.merged_trips_.at(lc.trips_)) {
      utl::verify(ci != nullptr, "get_trip_capacity: missing connection_info");

      auto tl_capacity = 0U;
      auto tl_capacity_src = capacity_source::SPECIAL;
      auto const trip_capacity =
          get_trip_capacity(sched, caps, trp, ci, lc.full_con_->clasz_);
      if (trip_capacity) {
        tl_capacity = trip_capacity->first;
        tl_capacity_src = trip_capacity->second;
      }

      auto tf_capacity = vehicle_capacity{};
      auto tf_found = false;
      auto tf_all_vehicles_found = false;
      auto vehicles = std::vector<Offset<PaxMonVehicleCapacityInfo>>{};
      auto vehicle_groups = std::vector<Offset<PaxMonVehicleGroupInfo>>{};
      auto const* tf_sec =
          get_trip_formation_section(sched, caps, trp, sec.ev_key_from());
      if (tf_sec != nullptr) {
        tf_found = true;
        tf_all_vehicles_found = true;

        vehicle_groups =
            utl::to_vec(tf_sec->vehicle_groups_, [&](vehicle_group const& vg) {
              return CreatePaxMonVehicleGroupInfo(
                  mc, mc.CreateSharedString(vg.name_.str()),
                  to_optional_fbs_station(mc, sched, vg.start_eva_),
                  to_optional_fbs_station(mc, sched, vg.destination_eva_),
                  mc.CreateSharedString(boost::uuids::to_string(vg.trip_uuid_)),
                  to_fbs(mc, vg.primary_trip_id_));
            });

        vehicles.reserve(tf_sec->vehicles_.size());
        for (auto const& vi : tf_sec->vehicles_) {
          auto const vgs = mc.CreateVector(utl::to_vec(
              vi.vehicle_groups_, [](auto const& vg) { return vg; }));
          if (auto const it = caps.vehicle_capacity_map_.find(vi.uic_);
              it != end(caps.vehicle_capacity_map_)) {
            auto const& vc = it->second;
            tf_capacity += vc;
            vehicles.emplace_back(CreatePaxMonVehicleCapacityInfo(
                mc, vi.uic_, true, mc.CreateSharedString(vi.baureihe_.str()),
                mc.CreateSharedString(vi.type_code_.str()),
                mc.CreateSharedString(vi.order_.str()),
                to_fbs_capacity_data(mc, vc), vgs));
          } else {
            tf_all_vehicles_found = false;
            vehicles.emplace_back(CreatePaxMonVehicleCapacityInfo(
                mc, vi.uic_, false, mc.CreateSharedString(vi.baureihe_.str()),
                mc.CreateSharedString(vi.type_code_.str()),
                mc.CreateSharedString(vi.order_.str()),
                to_fbs_capacity_data(mc, vehicle_capacity{}), vgs));
          }
        }
      }

      auto override = std::vector<Offset<PaxMonCapacityData>>{};
      if (auto const override_cap =
              get_override_capacity(sched, caps, trp, sec.ev_key_from());
          override_cap) {
        override.emplace_back(to_fbs_capacity_data(mc, *override_cap));
      }

      merged_trip_infos.emplace_back(CreatePaxMonMergedTripCapacityInfo(
          mc, to_fbs(sched, mc, trp),
          to_fbs(mc, get_service_info(sched, *lc.full_con_, ci)), tl_capacity,
          to_fbs_capacity_source(tl_capacity_src),
          to_fbs_capacity_data(mc, tf_capacity), tf_found,
          tf_all_vehicles_found, mc.CreateVector(vehicles),
          mc.CreateVector(vehicle_groups), mc.CreateVector(override)));

      ci = ci->merged_with_;
    }

    auto const lookup_result =
        get_capacity(sched, lc, sec.ev_key_from(), sec.ev_key_to(), caps);

    return CreatePaxMonSectionCapacityInfo(
        mc, to_fbs(mc, sec.from_station(sched)),
        to_fbs(mc, sec.to_station(sched)),
        motis_to_unixtime(sched, get_schedule_time(sched, sec.ev_key_from())),
        motis_to_unixtime(sched, sec.lcon().d_time_),
        motis_to_unixtime(sched, get_schedule_time(sched, sec.ev_key_to())),
        motis_to_unixtime(sched, sec.lcon().a_time_),
        lookup_result.first == UNKNOWN_CAPACITY ? PaxMonCapacityType_Unknown
                                                : PaxMonCapacityType_Known,
        lookup_result.first, to_fbs_capacity_source(lookup_result.second),
        mc.CreateVector(merged_trip_infos));
  };

  auto const trip_to_fbs = [&](trip const* trp) {
    return CreatePaxMonTripCapacityInfo(
        mc, to_fbs_trip_service_info(mc, sched, trp),
        mc.CreateVector(utl::to_vec(access::sections{trp}, section_to_fbs)));
  };

  mc.create_and_finish(
      MsgContent_PaxMonGetTripCapacityResponse,
      CreatePaxMonGetTripCapacityResponse(
          mc, mc.CreateVector(utl::to_vec(trips, trip_to_fbs)),
          caps.min_capacity_, caps.fuzzy_match_max_time_diff_,
          caps.trip_capacity_map_.size(), caps.category_capacity_map_.size(),
          caps.vehicle_capacity_map_.size(), caps.trip_formation_map_.size(),
          caps.override_map_.size())
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
