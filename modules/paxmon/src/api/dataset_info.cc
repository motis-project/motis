#include "motis/paxmon/api/dataset_info.h"

#include "utl/to_vec.h"

#include "motis/core/access/time_access.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr dataset_info(paxmon_data& data, schedule const& sched) {
  message_creator mc;

  mc.create_and_finish(
      MsgContent_PaxMonDatasetInfoResponse,
      CreatePaxMonDatasetInfoResponse(
          mc,
          mc.CreateVector(utl::to_vec(
              data.loaded_journey_files_,
              [&](auto const& ljf) {
                return CreatePaxMonJourneyFileInfo(
                    mc, mc.CreateString(ljf.path_.filename().string()),
                    ljf.last_modified_, ljf.matched_journeys_,
                    ljf.unmatched_journeys_, ljf.unmatched_journeys_rerouted_,
                    ljf.matched_groups_, ljf.unmatched_groups_,
                    ljf.unmatched_groups_rerouted_, ljf.matched_pax_,
                    ljf.unmatched_pax_, ljf.unmatched_pax_rerouted_);
              })),
          mc.CreateVector(utl::to_vec(
              data.loaded_capacity_files_,
              [&](auto const& lcf) {
                return CreatePaxMonCapacityFileInfo(
                    mc, mc.CreateString(lcf.path_.filename().string()),
                    lcf.last_modified_,
                    static_cast<PaxMonCapacityFileFormat>(lcf.format_),
                    lcf.loaded_entry_count_, lcf.skipped_entry_count_,
                    lcf.station_not_found_count_);
              })),
          CreatePaxMonScheduleInfo(
              mc,
              mc.CreateVector(utl::to_vec(sched.names_,
                                          [&](auto const& name) {
                                            return mc.CreateString(name.str());
                                          })),
              external_schedule_begin(sched), external_schedule_end(sched),
              sched.schedule_begin_, sched.schedule_end_,
              sched.stations_.size(), sched.trip_mem_.size(),
              sched.expanded_trips_.element_count()),
          data.motis_start_time_)
          .Union());

  return make_msg(mc);
}

}  // namespace motis::paxmon::api
