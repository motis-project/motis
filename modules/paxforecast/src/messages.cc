#include "motis/paxforecast/messages.h"

#include "utl/to_vec.h"

#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/messages.h"

#include "motis/paxforecast/error.h"

using namespace motis::module;
using namespace flatbuffers;
using namespace motis::paxmon;

namespace motis::paxforecast {

Offset<PaxForecastGroup> get_passenger_group_forecast(
    FlatBufferBuilder& fbb, schedule const& sched, passenger_group const& grp,
    group_simulation_result const& group_result) {
  return CreatePaxForecastGroup(
      fbb, to_fbs(sched, fbb, grp),
      fbs_localization_type(*group_result.localization_),
      to_fbs(sched, fbb, *group_result.localization_),
      fbb.CreateVector(
          utl::to_vec(group_result.alternatives_, [&](auto const& alt) {
            return CreatePaxForecastAlternative(
                fbb, to_fbs(sched, fbb, alt.first->compact_journey_),
                alt.second);
          })));
}

Offset<Vector<Offset<PaxMonTripLoadInfo>>> to_fbs(FlatBufferBuilder& fbb,
                                                  schedule const& sched,
                                                  universe const& uv,
                                                  load_forecast const& lfc) {
  return fbb.CreateVector(utl::to_vec(lfc.trips_, [&](auto const& tfc) {
    return to_fbs(fbb, sched, uv, tfc);
  }));
}

msg_ptr make_forecast_update_msg(schedule const& sched, universe const& uv,
                                 simulation_result const& sim_result,
                                 load_forecast const& lfc) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_PaxForecastUpdate,
      CreatePaxForecastUpdate(fbb, uv.id_, sched.system_time_,
                              fbb.CreateVector(utl::to_vec(
                                  sim_result.group_results_,
                                  [&](auto const& entry) {
                                    return get_passenger_group_forecast(
                                        fbb, sched, *entry.first, entry.second);
                                  })),
                              to_fbs(fbb, sched, uv, lfc))
          .Union(),
      "/paxforecast/passenger_forecast");
  return make_msg(fbb);
}

std::uint32_t get_station_index(schedule const& sched, String const* eva) {
  return get_station(sched, {eva->c_str(), eva->Length()})->index_;
}

measures::recipients from_fbs(schedule const& sched,
                              MeasureRecipients const* r) {
  return {utl::to_vec(*r->trips(),
                      [&](TripId const* t) { return to_extern_trip(t); }),
          utl::to_vec(*r->stations(), [&](String const* eva) {
            return get_station_index(sched, eva);
          })};
}

measures::trip_recommendation from_fbs(schedule const& sched,
                                       TripRecommendationMeasure const* m) {
  return {from_fbs(sched, m->recipients()),
          unix_to_motistime(sched.schedule_begin_, m->time()),
          utl::to_vec(*m->planned_trips(),
                      [&](TripId const* t) { return to_extern_trip(t); }),
          utl::to_vec(
              *m->planned_destinations(),
              [&](String const* eva) { return get_station_index(sched, eva); }),
          to_extern_trip(m->recommended_trip())};
}

measures::trip_load_information from_fbs(schedule const& sched,
                                         TripLoadInfoMeasure const* m) {
  return {from_fbs(sched, m->recipients()),
          unix_to_motistime(sched.schedule_begin_, m->time()),
          to_extern_trip(m->trip()),
          static_cast<measures::load_level>(m->level())};
}

measures::trip_with_load_level from_fbs(TripWithLoadLevel const* tll) {
  return {to_extern_trip(tll->trip()),
          static_cast<measures::load_level>(tll->level())};
}

measures::trip_load_recommendation from_fbs(
    schedule const& sched, TripLoadRecommendationMeasure const* m) {
  return {
      from_fbs(sched, m->recipients()),
      unix_to_motistime(sched.schedule_begin_, m->time()),
      utl::to_vec(
          *m->planned_destinations(),
          [&](String const* eva) { return get_station_index(sched, eva); }),
      utl::to_vec(*m->full_trips(),
                  [&](TripWithLoadLevel const* tll) { return from_fbs(tll); }),
      utl::to_vec(*m->recommended_trips(),
                  [&](TripWithLoadLevel const* tll) { return from_fbs(tll); }),
  };
}

measures::rt_update from_fbs(schedule const& sched, RtUpdateMeasure const* m) {
  return {from_fbs(sched, m->recipients()),
          unix_to_motistime(sched.schedule_begin_, m->time()), m->type(),
          m->content()->str()};
}

measures::measure_collection from_fbs(
    schedule const& sched, Vector<Offset<MeasureWrapper>> const* ms) {
  measures::measure_collection res;
  for (auto const* fm : *ms) {
    switch (fm->measure_type()) {
      case Measure_TripRecommendationMeasure: {
        auto const m = from_fbs(
            sched,
            reinterpret_cast<TripRecommendationMeasure const*>(fm->measure()));
        res[m.time_].emplace_back(m);
        break;
      }
      case Measure_TripLoadInfoMeasure: {
        auto const m = from_fbs(
            sched, reinterpret_cast<TripLoadInfoMeasure const*>(fm->measure()));
        res[m.time_].emplace_back(m);
        break;
      }
      case Measure_TripLoadRecommendationMeasure: {
        auto const m = from_fbs(
            sched, reinterpret_cast<TripLoadRecommendationMeasure const*>(
                       fm->measure()));
        res[m.time_].emplace_back(m);
        break;
      }
      case Measure_RtUpdateMeasure: {
        auto const m = from_fbs(
            sched, reinterpret_cast<RtUpdateMeasure const*>(fm->measure()));
        res[m.time_].emplace_back(m);
        break;
      }
      default: throw std::system_error{error::unsupported_measure};
    }
  }
  return res;
}

}  // namespace motis::paxforecast
