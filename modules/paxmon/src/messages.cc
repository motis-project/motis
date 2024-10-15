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

inline std::uint64_t to_fbs_time(schedule const& sched, time const t) {
  return t != INVALID_TIME ? static_cast<std::uint64_t>(
                                 motis_to_unixtime(sched.schedule_begin_, t))
                           : 0ULL;
}

inline time from_fbs_time(schedule const& sched, std::uint64_t const ut) {
  return ut != 0ULL ? unix_to_motistime(sched.schedule_begin_, ut)
                    : INVALID_TIME;
}

Offset<PaxMonTransferInfo> to_fbs(FlatBufferBuilder& fbb,
                                  std::optional<transfer_info> const& ti) {
  if (ti) {
    auto const& val = ti.value();
    return CreatePaxMonTransferInfo(
        fbb,
        val.type_ == transfer_info::type::SAME_STATION
            ? PaxMonTransferType_SAME_STATION
            : PaxMonTransferType_FOOTPATH,
        val.duration_);
  } else {
    return CreatePaxMonTransferInfo(fbb, PaxMonTransferType_NONE);
  }
}

std::optional<transfer_info> from_fbs(PaxMonTransferInfo const* ti) {
  switch (ti->type()) {
    case PaxMonTransferType_SAME_STATION:
      return transfer_info{static_cast<duration>(ti->duration()),
                           transfer_info::type::SAME_STATION};
    case PaxMonTransferType_FOOTPATH:
      return transfer_info{static_cast<duration>(ti->duration()),
                           transfer_info::type::FOOTPATH};
    default: return {};
  }
}

Offset<PaxMonCompactJourneyLeg> to_fbs(schedule const& sched,
                                       FlatBufferBuilder& fbb,
                                       journey_leg const& leg) {
  return CreatePaxMonCompactJourneyLeg(
      fbb, to_fbs_trip_service_info(fbb, sched, leg),
      to_fbs(fbb, *sched.stations_[leg.enter_station_id_]),
      to_fbs(fbb, *sched.stations_[leg.exit_station_id_]),
      motis_to_unixtime(sched, leg.enter_time_),
      motis_to_unixtime(sched, leg.exit_time_),
      to_fbs(fbb, leg.enter_transfer_));
}

journey_leg from_fbs(schedule const& sched,
                     PaxMonCompactJourneyLeg const* leg) {
  return {get_trip(sched, to_extern_trip(leg->trip()->trip()))->trip_idx_,
          get_station(sched, leg->enter_station()->id()->str())->index_,
          get_station(sched, leg->exit_station()->id()->str())->index_,
          unix_to_motistime(sched, leg->enter_time()),
          unix_to_motistime(sched, leg->exit_time()),
          from_fbs(leg->enter_transfer())};
}

Offset<PaxMonCompactJourney> to_fbs(schedule const& sched,
                                    FlatBufferBuilder& fbb,
                                    compact_journey const& cj) {
  return CreatePaxMonCompactJourney(
      fbb, fbb.CreateVector(utl::to_vec(cj.legs_, [&](journey_leg const& leg) {
        return to_fbs(sched, fbb, leg);
      })));
}

compact_journey from_fbs(schedule const& sched,
                         PaxMonCompactJourney const* cj) {
  return {utl::to_vec(*cj->legs(),
                      [&](auto const& leg) { return from_fbs(sched, leg); })};
}

Offset<PaxMonDataSource> to_fbs(FlatBufferBuilder& fbb, data_source const& ds) {
  return CreatePaxMonDataSource(fbb, ds.primary_ref_, ds.secondary_ref_);
}

data_source from_fbs(PaxMonDataSource const* ds) {
  return {ds->primary_ref(), ds->secondary_ref()};
}

Offset<PaxMonGroup> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                           passenger_group const& pg) {
  return CreatePaxMonGroup(
      fbb, pg.id_, to_fbs(fbb, pg.source_), pg.passengers_,
      to_fbs(sched, fbb, pg.compact_planned_journey_), pg.probability_,
      to_fbs_time(sched, pg.planned_arrival_time_),
      static_cast<std::underlying_type_t<group_source_flags>>(pg.source_flags_),
      pg.generation_, pg.previous_version_, to_fbs_time(sched, pg.added_time_),
      pg.estimated_delay());
}

passenger_group from_fbs(schedule const& sched, PaxMonGroup const* pg) {
  return make_passenger_group(
      from_fbs(sched, pg->planned_journey()), from_fbs(pg->source()),
      static_cast<std::uint16_t>(pg->passenger_count()),
      from_fbs_time(sched, pg->planned_arrival_time()),
      static_cast<group_source_flags>(pg->source_flags()), pg->probability(),
      from_fbs_time(sched, pg->added_time()), pg->previous_version(),
      pg->generation(), pg->estimated_delay(), pg->id());
}

PaxMonGroupBaseInfo to_fbs_base_info(FlatBufferBuilder& /*fbb*/,
                                     passenger_group const& pg) {
  return PaxMonGroupBaseInfo{pg.id_, pg.passengers_, pg.probability_};
}

Offset<void> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                    passenger_localization const& loc) {
  if (loc.in_trip()) {
    return CreatePaxMonInTrip(
               fbb, to_fbs(sched, fbb, loc.in_trip_),
               to_fbs(fbb, *loc.at_station_),
               motis_to_unixtime(sched, loc.schedule_arrival_time_),
               motis_to_unixtime(sched, loc.current_arrival_time_))
        .Union();
  } else {
    return CreatePaxMonAtStation(
               fbb, to_fbs(fbb, *loc.at_station_),
               motis_to_unixtime(sched, loc.schedule_arrival_time_),
               motis_to_unixtime(sched, loc.current_arrival_time_),
               loc.first_station_)
        .Union();
  }
}

passenger_localization from_fbs(schedule const& sched,
                                PaxMonLocalization const loc_type,
                                void const* loc_ptr) {
  // NOTE: remaining_interchanges_ is currently not included in messages
  switch (loc_type) {
    case PaxMonLocalization_PaxMonInTrip: {
      auto const loc = reinterpret_cast<PaxMonInTrip const*>(loc_ptr);
      return {from_fbs(sched, loc->trip()),
              get_station(sched, loc->next_station()->id()->str()),
              unix_to_motistime(sched, loc->schedule_arrival_time()),
              unix_to_motistime(sched, loc->current_arrival_time()),
              false,
              {}};
    }
    case PaxMonLocalization_PaxMonAtStation: {
      auto const loc = reinterpret_cast<PaxMonAtStation const*>(loc_ptr);
      return {nullptr,
              get_station(sched, loc->station()->id()->str()),
              unix_to_motistime(sched, loc->schedule_arrival_time()),
              unix_to_motistime(sched, loc->current_arrival_time()),
              loc->first_station(),
              {}};
    }
    default:
      throw utl::fail("invalid passenger localization type: {}", loc_type);
  }
}

passenger_localization from_fbs(schedule const& sched,
                                PaxMonLocalizationWrapper const* loc_wrapper) {
  return from_fbs(sched, loc_wrapper->localization_type(),
                  loc_wrapper->localization());
}

PaxMonLocalization fbs_localization_type(passenger_localization const& loc) {
  return loc.in_trip() ? PaxMonLocalization_PaxMonInTrip
                       : PaxMonLocalization_PaxMonAtStation;
}

Offset<PaxMonLocalizationWrapper> to_fbs_localization_wrapper(
    schedule const& sched, FlatBufferBuilder& fbb,
    passenger_localization const& loc) {
  return CreatePaxMonLocalizationWrapper(fbb, fbs_localization_type(loc),
                                         to_fbs(sched, fbb, loc));
}

Offset<PaxMonEvent> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                           monitoring_event const& me) {
  return CreatePaxMonEvent(
      fbb, static_cast<PaxMonEventType>(me.type_),
      to_fbs(sched, fbb, me.group_), fbs_localization_type(me.localization_),
      to_fbs(sched, fbb, me.localization_),
      static_cast<PaxMonReachabilityStatus>(me.reachability_status_),
      to_fbs_time(sched, me.expected_arrival_time_));
}

Offset<Vector<PaxMonPdfEntry const*>> pdf_to_fbs(FlatBufferBuilder& fbb,
                                                 pax_pdf const& pdf) {
  auto entries = std::vector<PaxMonPdfEntry>{};
  for (auto const& [pax, prob] : utl::enumerate(pdf.data_)) {
    if (prob != 0.F) {
      entries.emplace_back(static_cast<std::uint32_t>(pax), prob);
    }
  }
  return fbb.CreateVectorOfStructs(entries);
}

Offset<Vector<PaxMonCdfEntry const*>> cdf_to_fbs(FlatBufferBuilder& fbb,
                                                 pax_cdf const& cdf) {
  auto entries = std::vector<PaxMonCdfEntry>{};
  entries.reserve(cdf.data_.size());
  if (!cdf.data_.empty()) {
    auto last_prob = 0.0F;
    auto const last_index = cdf.data_.size() - 1;
    for (auto const& [pax, prob] : utl::enumerate(cdf.data_)) {
      if (prob != last_prob || pax == last_index) {
        entries.emplace_back(static_cast<std::uint32_t>(pax), prob);
        last_prob = prob;
      }
    }
  }
  return fbb.CreateVectorOfStructs(entries);
}

PaxMonCapacityType get_capacity_type(motis::paxmon::edge const* e) {
  if (e->has_unknown_capacity()) {
    return PaxMonCapacityType_Unknown;
  } else if (e->has_unlimited_capacity()) {
    return PaxMonCapacityType_Unlimited;
  } else {
    return PaxMonCapacityType_Known;
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

Offset<TripServiceInfo> to_fbs_trip_service_info(FlatBufferBuilder& fbb,
                                                 schedule const& sched,
                                                 journey_leg const& leg) {
  return to_fbs_trip_service_info(fbb, sched, get_trip(sched, leg.trip_idx_),
                                  get_service_infos_for_leg(sched, leg));
}

Offset<PaxMonDistribution> to_fbs_distribution(FlatBufferBuilder& fbb,
                                               pax_pdf const& pdf,
                                               pax_stats const& stats) {
  return CreatePaxMonDistribution(fbb, stats.limits_.min_, stats.limits_.max_,
                                  stats.q5_, stats.q50_, stats.q95_,
                                  pdf_to_fbs(fbb, pdf));
}

Offset<PaxMonDistribution> to_fbs_distribution(FlatBufferBuilder& fbb,
                                               pax_pdf const& pdf,
                                               pax_cdf const& cdf) {
  return to_fbs_distribution(fbb, pdf, get_pax_stats(cdf));
}

Offset<PaxMonEdgeLoadInfo> to_fbs(FlatBufferBuilder& fbb, schedule const& sched,
                                  universe const& uv,
                                  edge_load_info const& eli) {
  auto const from = eli.edge_->from(uv);
  auto const to = eli.edge_->to(uv);
  return CreatePaxMonEdgeLoadInfo(
      fbb, to_fbs(fbb, from->get_station(sched)),
      to_fbs(fbb, to->get_station(sched)),
      motis_to_unixtime(sched, from->schedule_time()),
      motis_to_unixtime(sched, from->current_time()),
      motis_to_unixtime(sched, to->schedule_time()),
      motis_to_unixtime(sched, to->current_time()),
      get_capacity_type(eli.edge_), eli.edge_->capacity(),
      to_fbs_distribution(fbb, eli.forecast_pdf_, eli.forecast_cdf_),
      eli.updated_, eli.possibly_over_capacity_, eli.probability_over_capacity_,
      eli.expected_passengers_);
}

Offset<PaxMonTripLoadInfo> to_fbs(FlatBufferBuilder& fbb, schedule const& sched,
                                  universe const& uv,
                                  trip_load_info const& tli) {
  return CreatePaxMonTripLoadInfo(
      fbb, to_fbs_trip_service_info(fbb, sched, tli.trp_),
      fbb.CreateVector(utl::to_vec(tli.edges_, [&](auto const& efc) {
        return to_fbs(fbb, sched, uv, efc);
      })));
}

}  // namespace motis::paxmon
