#include "motis/paxmon/messages.h"

#include <cassert>
#include <algorithm>

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/access/station_access.h"
#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

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
      fbb, to_fbs(fbb, leg.trip_),
      to_fbs(fbb, *sched.stations_[leg.enter_station_id_]),
      to_fbs(fbb, *sched.stations_[leg.exit_station_id_]),
      motis_to_unixtime(sched, leg.enter_time_),
      motis_to_unixtime(sched, leg.exit_time_),
      to_fbs(fbb, leg.enter_transfer_));
}

journey_leg from_fbs(schedule const& sched, CompactJourneyLeg const* leg) {
  return {to_extern_trip(leg->trip()),
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
  return CreatePassengerGroup(fbb, pg.id_, to_fbs(fbb, pg.source_),
                              pg.passengers_,
                              to_fbs(sched, fbb, pg.compact_planned_journey_));
}

Offset<void> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                    passenger_localization const& loc) {
  if (loc.in_trip()) {
    return CreatePassengerInTrip(fbb, to_fbs(sched, fbb, loc.in_trip_),
                                 to_fbs(fbb, *loc.at_station_),
                                 motis_to_unixtime(sched, loc.arrival_time_))
        .Union();
  } else {
    return CreatePassengerAtStation(fbb, to_fbs(fbb, *loc.at_station_),
                                    motis_to_unixtime(sched, loc.arrival_time_),
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
              unix_to_motistime(sched, loc->arrival_time()), false};
    }
    case PassengerLocalization_PassengerAtStation: {
      auto const loc = reinterpret_cast<PassengerAtStation const*>(loc_ptr);
      return {nullptr, get_station(sched, loc->station()->id()->str()),
              unix_to_motistime(sched, loc->arrival_time()),
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

}  // namespace motis::paxmon
