#include "motis/paxforecast/messages.h"

#include "utl/enumerate.h"
#include "utl/to_vec.h"

#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace flatbuffers;
using namespace motis::paxmon;

namespace motis::paxforecast {

Offset<PassengerGroupForecast> get_passenger_group_forecast(
    FlatBufferBuilder& fbb, schedule const& sched, passenger_group const& grp,
    group_simulation_result const& group_result) {
  return CreatePassengerGroupForecast(
      fbb, to_fbs(sched, fbb, grp),
      fbs_localization_type(*group_result.localization_),
      to_fbs(sched, fbb, *group_result.localization_),
      fbb.CreateVector(
          utl::to_vec(group_result.alternatives_, [&](auto const& alt) {
            return CreateForecastAlternative(
                fbb, to_fbs(sched, fbb, alt.first->compact_journey_),
                alt.second);
          })));
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

Offset<EdgeForecast> to_fbs(FlatBufferBuilder& fbb, schedule const& sched,
                            graph const& g, edge_forecast const& efc) {
  auto const from = efc.edge_->from(g);
  auto const to = efc.edge_->to(g);
  return CreateEdgeForecast(fbb, to_fbs(fbb, from->get_station(sched)),
                            to_fbs(fbb, to->get_station(sched)),
                            motis_to_unixtime(sched, from->schedule_time()),
                            motis_to_unixtime(sched, to->schedule_time()),
                            get_capacity_type(efc.edge_), efc.edge_->capacity(),
                            cdf_to_fbs(fbb, efc.forecast_cdf_), efc.updated_,
                            efc.possibly_over_capacity_);
}

Offset<TripForecast> to_fbs(FlatBufferBuilder& fbb, schedule const& sched,
                            graph const& g, trip_forecast const& tfc) {
  return CreateTripForecast(
      fbb, to_fbs(sched, fbb, tfc.trp_),
      fbb.CreateVector(utl::to_vec(tfc.edges_, [&](auto const& efc) {
        return to_fbs(fbb, sched, g, efc);
      })));
}

Offset<Vector<Offset<TripForecast>>> to_fbs(FlatBufferBuilder& fbb,
                                            schedule const& sched,
                                            graph const& g,
                                            load_forecast const& lfc) {
  return fbb.CreateVector(utl::to_vec(
      lfc.trips_, [&](auto const& tfc) { return to_fbs(fbb, sched, g, tfc); }));
}

msg_ptr make_passenger_forecast_msg(schedule const& sched,
                                    motis::paxmon::paxmon_data const& data,
                                    simulation_result const& sim_result,
                                    load_forecast const& lfc) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_PassengerForecast,
      CreatePassengerForecast(fbb, sched.system_time_,
                              fbb.CreateVector(utl::to_vec(
                                  sim_result.group_results_,
                                  [&](auto const& entry) {
                                    return get_passenger_group_forecast(
                                        fbb, sched, *entry.first, entry.second);
                                  })),
                              to_fbs(fbb, sched, data.graph_, lfc))
          .Union(),
      "/paxforecast/passenger_forecast");
  return make_msg(fbb);
}

}  // namespace motis::paxforecast
