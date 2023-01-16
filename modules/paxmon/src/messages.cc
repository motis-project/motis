#include "motis/paxmon/messages.h"

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

inline PaxMonTransferType to_fbs_transfer_type(transfer_info::type const t) {
  switch (t) {
    case transfer_info::type::SAME_STATION:
      return PaxMonTransferType_SAME_STATION;
    case transfer_info::type::FOOTPATH: return PaxMonTransferType_FOOTPATH;
    case transfer_info::type::MERGE: return PaxMonTransferType_MERGE;
    case transfer_info::type::THROUGH: return PaxMonTransferType_THROUGH;
  }
  return PaxMonTransferType_NONE;
}

Offset<PaxMonTransferInfo> to_fbs(FlatBufferBuilder& fbb,
                                  std::optional<transfer_info> const& ti) {
  if (ti) {
    auto const& val = ti.value();
    return CreatePaxMonTransferInfo(fbb, to_fbs_transfer_type(val.type_),
                                    val.duration_);
  } else {
    return CreatePaxMonTransferInfo(fbb, PaxMonTransferType_NONE);
  }
}

std::optional<transfer_info> from_fbs(PaxMonTransferInfo const* ti) {
  auto const dur = static_cast<duration>(ti->duration());
  switch (ti->type()) {
    case PaxMonTransferType_SAME_STATION:
      return transfer_info{dur, transfer_info::type::SAME_STATION};
    case PaxMonTransferType_FOOTPATH:
      return transfer_info{dur, transfer_info::type::FOOTPATH};
    case PaxMonTransferType_MERGE:
      return transfer_info{dur, transfer_info::type::MERGE};
    case PaxMonTransferType_THROUGH:
      return transfer_info{dur, transfer_info::type::THROUGH};
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
      fbb, fbb.CreateVector(utl::to_vec(cj.legs(), [&](journey_leg const& leg) {
        return to_fbs(sched, fbb, leg);
      })));
}

Offset<PaxMonCompactJourney> to_fbs(schedule const& sched,
                                    FlatBufferBuilder& fbb,
                                    fws_compact_journey const& cj) {
  return CreatePaxMonCompactJourney(
      fbb, fbb.CreateVector(utl::to_vec(cj.legs(), [&](journey_leg const& leg) {
        return to_fbs(sched, fbb, leg);
      })));
}

compact_journey from_fbs(schedule const& sched,
                         PaxMonCompactJourney const* cj) {
  return compact_journey{utl::to_vec(
      *cj->legs(), [&](auto const& leg) { return from_fbs(sched, leg); })};
}

Offset<PaxMonDataSource> to_fbs(FlatBufferBuilder& fbb, data_source const& ds) {
  return CreatePaxMonDataSource(fbb, ds.primary_ref_, ds.secondary_ref_);
}

data_source from_fbs(PaxMonDataSource const* ds) {
  return {ds->primary_ref(), ds->secondary_ref()};
}

Offset<PaxMonGroupRoute> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                                temp_group_route const& tgr) {
  return CreatePaxMonGroupRoute(
      fbb, tgr.index_.has_value() ? tgr.index_.value() : -1,
      to_fbs(sched, fbb, tgr.journey_), tgr.probability_,
      to_fbs_time(sched, tgr.planned_arrival_time_), tgr.estimated_delay_,
      static_cast<std::uint8_t>(tgr.source_flags_), tgr.planned_,
      false /* broken */, false /* disabled */,
      false /* destination_unreachable */);
}

Offset<PaxMonGroupRoute> to_fbs(schedule const& sched,
                                passenger_group_container const& pgc,
                                FlatBufferBuilder& fbb, group_route const& gr) {
  return CreatePaxMonGroupRoute(
      fbb, gr.local_group_route_index_,
      to_fbs(sched, fbb, pgc.journey(gr.compact_journey_index_)),
      gr.probability_, to_fbs_time(sched, gr.planned_arrival_time_),
      gr.estimated_delay_, static_cast<std::uint8_t>(gr.source_flags_),
      gr.planned_, gr.broken_, gr.disabled_, gr.destination_unreachable_);
}

temp_group_route from_fbs(schedule const& sched, PaxMonGroupRoute const* gr) {
  return temp_group_route{
      gr->index() >= 0 ? std::optional<local_group_route_index>{static_cast<
                             local_group_route_index>(gr->index())}
                       : std::nullopt,
      gr->probability(),
      from_fbs(sched, gr->journey()),
      from_fbs_time(sched, gr->planned_arrival_time()),
      gr->estimated_delay(),
      static_cast<route_source_flags>(gr->source_flags()),
      gr->planned()};
}

std::optional<broken_transfer_info> from_fbs(
    schedule const& sched,
    Vector<Offset<PaxMonBrokenTransferInfo>> const* opt) {
  if (opt->size() == 1) {
    auto const* bti = opt->Get(0);
    return {broken_transfer_info{
        bti->leg_index(), static_cast<transfer_direction_t>(bti->direction()),
        from_fbs_time(sched, bti->current_arrival_time()),
        from_fbs_time(sched, bti->current_departure_time()),
        bti->required_transfer_time(), bti->arrival_canceled(),
        bti->departure_canceled()}};
  } else if (opt->size() == 0) {
    return {};
  } else {
    throw utl::fail(
        "invalid optional PaxMonBrokenTransferInfo: {} entries (expected 0 or "
        "1)",
        opt->size());
  }
}

Offset<Vector<Offset<PaxMonBrokenTransferInfo>>> broken_transfer_info_to_fbs(
    FlatBufferBuilder& fbb, schedule const& sched,
    std::optional<broken_transfer_info> const& opt) {
  if (opt.has_value()) {
    auto const& bti = opt.value();
    return fbb.CreateVector(std::vector<Offset<PaxMonBrokenTransferInfo>>{
        CreatePaxMonBrokenTransferInfo(
            fbb, bti.leg_index_,
            static_cast<PaxMonTransferDirection>(bti.direction_),
            to_fbs_time(sched, bti.current_arrival_time_),
            to_fbs_time(sched, bti.current_departure_time_),
            bti.required_transfer_time_, bti.arrival_canceled_,
            bti.departure_canceled_)});
  } else {
    return fbb.CreateVector(std::vector<Offset<PaxMonBrokenTransferInfo>>{});
  }
}

Offset<PaxMonRerouteLogRoute> to_fbs(FlatBufferBuilder& fbb,
                                     reroute_log_route_info const& ri) {
  return CreatePaxMonRerouteLogRoute(fbb, ri.route_, ri.previous_probability_,
                                     ri.new_probability_);
}

Offset<PaxMonRerouteLogEntry> to_fbs(schedule const& sched,
                                     FlatBufferBuilder& fbb,
                                     passenger_group_container const& pgc,
                                     reroute_log_entry const& entry) {
  return CreatePaxMonRerouteLogEntry(
      fbb, entry.system_time_, entry.reroute_time_,
      static_cast<PaxMonRerouteReason>(entry.reason_),
      broken_transfer_info_to_fbs(fbb, sched, entry.broken_transfer_),
      to_fbs(fbb, entry.old_route_),
      fbb.CreateVector(utl::to_vec(
          pgc.log_entry_new_routes_.at(entry.index_),
          [&](auto const& new_route) { return to_fbs(fbb, new_route); })),
      fbs_localization_type(entry.localization_),
      to_fbs(sched, fbb, entry.localization_));
}

Offset<PaxMonGroup> to_fbs(schedule const& sched,
                           passenger_group_container const& pgc,
                           FlatBufferBuilder& fbb, passenger_group const& pg,
                           bool const with_reroute_log) {
  return CreatePaxMonGroup(
      fbb, pg.id_, to_fbs(fbb, pg.source_), pg.passengers_,
      fbb.CreateVector(utl::to_vec(
          pgc.routes(pg.id_),
          [&](group_route const& gr) { return to_fbs(sched, pgc, fbb, gr); })),
      fbb.CreateVector(with_reroute_log
                           ? utl::to_vec(pgc.reroute_log_entries(pg.id_),
                                         [&](auto const& entry) {
                                           return to_fbs(sched, fbb, pgc,
                                                         entry);
                                         })
                           : std::vector<Offset<PaxMonRerouteLogEntry>>{}));
}

temp_passenger_group from_fbs(schedule const& sched, PaxMonGroup const* pg) {
  return temp_passenger_group{
      pg->id(), from_fbs(pg->source()), pg->passenger_count(),
      utl::to_vec(*pg->routes(), [&](PaxMonGroupRoute const* gr) {
        return from_fbs(sched, gr);
      })};
}

temp_passenger_group_with_route from_fbs(schedule const& sched,
                                         PaxMonGroupWithRoute const* pgwr) {
  return temp_passenger_group_with_route{
      static_cast<passenger_group_index>(pgwr->group_id()),
      from_fbs(pgwr->source()), pgwr->passenger_count(),
      from_fbs(sched, pgwr->route())};
}

Offset<PaxMonGroupWithRoute> to_fbs(schedule const& sched,
                                    passenger_group_container const& pgc,
                                    FlatBufferBuilder& fbb,
                                    passenger_group_with_route const& pgwr) {
  auto const& pg = pgc.group(pgwr.pg_);
  auto const& gr = pgc.route(pgwr);
  return CreatePaxMonGroupWithRoute(fbb, pg.id_, to_fbs(fbb, pg.source_),
                                    pg.passengers_,
                                    to_fbs(sched, pgc, fbb, gr));
}

Offset<PaxMonGroupWithRoute> to_fbs(
    FlatBufferBuilder& fbb, temp_passenger_group_with_route const& tpgr) {
  return CreatePaxMonGroupWithRoute(
      fbb, tpgr.group_id_, to_fbs(fbb, tpgr.source_), tpgr.passengers_);
}

PaxMonGroupRouteBaseInfo to_fbs_base_info(FlatBufferBuilder& /*fbb*/,
                                          passenger_group const& pg,
                                          group_route const& gr) {
  return PaxMonGroupRouteBaseInfo{pg.id_, gr.local_group_route_index_,
                                  pg.passengers_, gr.probability_};
}

PaxMonGroupRouteBaseInfo to_fbs_base_info(
    FlatBufferBuilder& fbb, passenger_group_container const& pgc,
    passenger_group_with_route const& pgwr) {
  return to_fbs_base_info(fbb, pgc.group(pgwr.pg_), pgc.route(pgwr));
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

PaxMonLocalization fbs_localization_type(reroute_log_localization const& loc) {
  return loc.in_trip_ ? PaxMonLocalization_PaxMonInTrip
                      : PaxMonLocalization_PaxMonAtStation;
}

Offset<PaxMonLocalizationWrapper> to_fbs_localization_wrapper(
    schedule const& sched, FlatBufferBuilder& fbb,
    passenger_localization const& loc) {
  return CreatePaxMonLocalizationWrapper(fbb, fbs_localization_type(loc),
                                         to_fbs(sched, fbb, loc));
}

Offset<void> to_fbs(schedule const& sched, FlatBufferBuilder& fbb,
                    reroute_log_localization const& loc) {
  if (loc.in_trip_) {
    return CreatePaxMonInTrip(
               fbb, to_fbs(sched, fbb, get_trip(sched, loc.trip_idx_)),
               to_fbs(fbb, *sched.stations_.at(loc.station_id_)),
               motis_to_unixtime(sched, loc.schedule_arrival_time_),
               motis_to_unixtime(sched, loc.current_arrival_time_))
        .Union();
  } else {
    return CreatePaxMonAtStation(
               fbb, to_fbs(fbb, *sched.stations_.at(loc.station_id_)),
               motis_to_unixtime(sched, loc.schedule_arrival_time_),
               motis_to_unixtime(sched, loc.current_arrival_time_),
               loc.first_station_)
        .Union();
  }
}

Offset<PaxMonReachability> reachability_to_fbs(
    schedule const& sched, FlatBufferBuilder& fbb,
    reachability_status const status,
    std::optional<broken_transfer_info> const& bti) {
  return CreatePaxMonReachability(fbb,
                                  static_cast<PaxMonReachabilityStatus>(status),
                                  broken_transfer_info_to_fbs(fbb, sched, bti));
}

Offset<PaxMonEvent> to_fbs(schedule const& sched,
                           passenger_group_container const& pgc,
                           FlatBufferBuilder& fbb, monitoring_event const& me) {
  return CreatePaxMonEvent(
      fbb, static_cast<PaxMonEventType>(me.type_),
      to_fbs(sched, pgc, fbb, me.pgwr_),
      fbs_localization_type(me.localization_),
      to_fbs(sched, fbb, me.localization_),
      reachability_to_fbs(sched, fbb, me.reachability_status_,
                          me.broken_transfer_),
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
