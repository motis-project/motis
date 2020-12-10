#include "motis/paxforecast/messages.h"

#include "utl/to_vec.h"

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

Offset<Vector<Offset<TripLoadInfo>>> to_fbs(FlatBufferBuilder& fbb,
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
