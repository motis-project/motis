#include "motis/paxmon/messages.h"

#include <cassert>
#include <algorithm>

#include "utl/enumerate.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/get_load.h"

using namespace motis::module;
using namespace flatbuffers;

namespace motis::paxmon {

Offset<TransferInfo> to_fbs(FlatBufferBuilder& fbb,
                            std::optional<transfer_info> const& ti) {
  if (ti) {
    auto const& val = ti.value();
    return CreateTransferInfo(fbb,
                              val.type_ == transfer_info::type::SAME_STATION
                                  ? TransferType_SAME_STATION
                                  : TransferType_FOOTPATH,
                              val.duration_);
  } else {
    return CreateTransferInfo(fbb, TransferType_NONE);
  }
}

std::optional<transfer_info> from_fbs(TransferInfo const* ti) {
  switch (ti->type()) {
    case TransferType_SAME_STATION:
      return transfer_info{static_cast<duration>(ti->duration()),
                           transfer_info::type::SAME_STATION};
    case TransferType_FOOTPATH:
      return transfer_info{static_cast<duration>(ti->duration()),
                           transfer_info::type::FOOTPATH};
    default: return {};
  }
}

Offset<CompactJourneyLeg> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                                 journey_leg const& leg) {
  return CreateCompactJourneyLeg(
      fbb, to_fbs(fbb, to_extern_trip(sched, leg.trip_)),
      to_fbs(fbb, *sched.stations_[leg.enter_station_id_]),
      to_fbs(fbb, *sched.stations_[leg.exit_station_id_]),
      motis_to_unixtime(sched, leg.enter_time_),
      motis_to_unixtime(sched, leg.exit_time_),
      to_fbs(fbb, leg.enter_transfer_));
}

journey_leg from_fbs(schedule const& sched, CompactJourneyLeg const* leg) {
  return {get_trip(sched, to_extern_trip(leg->trip())),
          get_station(sched, leg->enter_station()->id()->str())->index_,
          get_station(sched, leg->exit_station()->id()->str())->index_,
          unix_to_motistime(sched, leg->enter_time()),
          unix_to_motistime(sched, leg->exit_time()),
          from_fbs(leg->enter_transfer())};
}

Offset<CompactJourney> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                              compact_journey const& cj) {
  return CreateCompactJourney(
      fbb, fbb.CreateVector(utl::to_vec(cj.legs_, [&](journey_leg const& leg) {
        return to_fbs(sched, fbb, leg);
      })));
}

compact_journey from_fbs(schedule const& sched, CompactJourney const* cj) {
  return {utl::to_vec(*cj->legs(),
                      [&](auto const& leg) { return from_fbs(sched, leg); })};
}

Offset<DataSource> to_fbs(FlatBufferBuilder& fbb, data_source const& ds) {
  return CreateDataSource(fbb, ds.primary_ref_, ds.secondary_ref_);
}

data_source from_fbs(DataSource const* ds) {
  return {ds->primary_ref(), ds->secondary_ref()};
}

Offset<PassengerGroup> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                              passenger_group const& pg) {
  return CreatePassengerGroup(
      fbb, pg.id_, to_fbs(fbb, pg.source_), pg.passengers_,
      to_fbs(sched, fbb, pg.compact_planned_journey_), pg.probability_,
      pg.planned_arrival_time_ != INVALID_TIME
          ? motis_to_unixtime(sched, pg.planned_arrival_time_)
          : 0,
      static_cast<std::underlying_type_t<group_source_flags>>(
          pg.source_flags_));
}

passenger_group from_fbs(schedule const& sched, PassengerGroup const* pg) {
  return passenger_group{
      from_fbs(sched, pg->planned_journey()),
      pg->id(),
      from_fbs(pg->source()),
      static_cast<std::uint16_t>(pg->passenger_count()),
      pg->planned_arrival_time() != 0
          ? unix_to_motistime(sched.schedule_begin_, pg->planned_arrival_time())
          : INVALID_TIME,
      static_cast<group_source_flags>(pg->source_flags()),
      true,
      pg->probability()};
}

Offset<void> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                    passenger_localization const& loc) {
  if (loc.in_trip()) {
    return CreatePassengerInTrip(
               fbb, to_fbs(sched, fbb, loc.in_trip_),
               to_fbs(fbb, *loc.at_station_),
               motis_to_unixtime(sched, loc.schedule_arrival_time_),
               motis_to_unixtime(sched, loc.current_arrival_time_))
        .Union();
  } else {
    return CreatePassengerAtStation(
               fbb, to_fbs(fbb, *loc.at_station_),
               motis_to_unixtime(sched, loc.schedule_arrival_time_),
               motis_to_unixtime(sched, loc.current_arrival_time_),
               loc.first_station_)
        .Union();
  }
}

passenger_localization from_fbs(schedule const& sched,
                                PassengerLocalization const loc_type,
                                void const* loc_ptr) {
  switch (loc_type) {
    case PassengerLocalization_PassengerInTrip: {
      auto const loc = reinterpret_cast<PassengerInTrip const*>(loc_ptr);
      return {from_fbs(sched, loc->trip()),
              get_station(sched, loc->next_station()->id()->str()),
              unix_to_motistime(sched, loc->schedule_arrival_time()),
              unix_to_motistime(sched, loc->current_arrival_time()), false};
    }
    case PassengerLocalization_PassengerAtStation: {
      auto const loc = reinterpret_cast<PassengerAtStation const*>(loc_ptr);
      return {nullptr, get_station(sched, loc->station()->id()->str()),
              unix_to_motistime(sched, loc->schedule_arrival_time()),
              unix_to_motistime(sched, loc->current_arrival_time()),
              loc->first_station()};
    }
    default:
      throw utl::fail("invalid passenger localization type: {}", loc_type);
  }
}

PassengerLocalization fbs_localization_type(passenger_localization const& loc) {
  return loc.in_trip() ? PassengerLocalization_PassengerInTrip
                       : PassengerLocalization_PassengerAtStation;
}

Offset<MonitoringEvent> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                               monitoring_event const& me) {
  return CreateMonitoringEvent(
      fbb, static_cast<MonitoringEventType>(me.type_),
      to_fbs(sched, fbb, me.group_), fbs_localization_type(me.localization_),
      to_fbs(sched, fbb, me.localization_),
      static_cast<ReachabilityStatus>(me.reachability_status_));
}

Offset<Vector<CdfEntry const*>> cdf_to_fbs(FlatBufferBuilder& fbb,
                                           pax_cdf const& cdf) {
  auto entries = std::vector<CdfEntry>{};
  entries.reserve(cdf.data_.size());
  auto last_prob = 0.0F;
  for (auto const& [pax, prob] : utl::enumerate(cdf.data_)) {
    if (prob != last_prob) {
      entries.emplace_back(pax, prob);
      last_prob = prob;
    }
  }
  return fbb.CreateVectorOfStructs(entries);
}

CapacityType get_capacity_type(motis::paxmon::edge const* e) {
  if (e->has_unknown_capacity()) {
    return CapacityType_Unknown;
  } else if (e->has_unlimited_capacity()) {
    return CapacityType_Unlimited;
  } else {
    return CapacityType_Known;
  }
}

Offset<ServiceInfo> to_fbs(FlatBufferBuilder& fbb, service_info const& si) {
  return CreateServiceInfo(
      fbb, fbb.CreateString(si.name_), fbb.CreateString(si.category_),
      si.train_nr_, fbb.CreateString(si.line_), fbb.CreateString(si.provider_),
      static_cast<service_class_t>(si.clasz_));
}

Offset<TripServiceInfo> to_fbs_trip_service_info(
    FlatBufferBuilder& fbb, schedule const& sched, trip const* trp,
    std::vector<std::pair<service_info, unsigned>> const& service_infos) {
  return CreateTripServiceInfo(
      fbb, to_fbs(sched, fbb, trp),
      to_fbs(fbb, *sched.stations_.at(trp->id_.primary_.get_station_id())),
      to_fbs(fbb, *sched.stations_.at(trp->id_.secondary_.target_station_id_)),
      fbb.CreateVector(utl::to_vec(service_infos, [&](auto const& sip) {
        return to_fbs(fbb, sip.first);
      })));
}

Offset<TripServiceInfo> to_fbs_trip_service_info(FlatBufferBuilder& fbb,
                                                 schedule const& sched,
                                                 trip const* trp) {
  return to_fbs_trip_service_info(fbb, sched, trp,
                                  get_service_infos(sched, trp));
}

Offset<EdgeLoadInfo> to_fbs(FlatBufferBuilder& fbb, schedule const& sched,
                            graph const& g, edge_load_info const& eli) {
  auto const from = eli.edge_->from(g);
  auto const to = eli.edge_->to(g);
  return CreateEdgeLoadInfo(fbb, to_fbs(fbb, from->get_station(sched)),
                            to_fbs(fbb, to->get_station(sched)),
                            motis_to_unixtime(sched, from->schedule_time()),
                            motis_to_unixtime(sched, from->current_time()),
                            motis_to_unixtime(sched, to->schedule_time()),
                            motis_to_unixtime(sched, to->current_time()),
                            get_capacity_type(eli.edge_), eli.edge_->capacity(),
                            cdf_to_fbs(fbb, eli.forecast_cdf_), eli.updated_,
                            eli.possibly_over_capacity_,
                            eli.expected_passengers_);
}

Offset<TripLoadInfo> to_fbs(FlatBufferBuilder& fbb, schedule const& sched,
                            graph const& g, trip_load_info const& tli) {
  return CreateTripLoadInfo(
      fbb, to_fbs_trip_service_info(fbb, sched, tli.trp_),
      fbb.CreateVector(utl::to_vec(tli.edges_, [&](auto const& efc) {
        return to_fbs(fbb, sched, g, efc);
      })));
}

}  // namespace motis::paxmon
