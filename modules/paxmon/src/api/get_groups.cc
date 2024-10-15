#include "motis/paxmon/api/get_status.h"

#include <vector>

#include "utl/verify.h"

#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/reachability.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr get_groups(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonGetGroupsRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;
  auto const all_generations = req->all_generations();
  auto const include_localization = req->include_localization();

  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  auto const search_time =
      static_cast<time>(current_time + req->preparation_time());
  if (include_localization) {
    utl::verify(current_time != INVALID_TIME, "invalid current system time");
  }

  message_creator mc;
  std::vector<flatbuffers::Offset<PaxMonGroup>> groups;
  std::vector<flatbuffers::Offset<PaxMonLocalizationWrapper>> localizations;

  auto const add_by_data_source = [&](data_source const& ds) {
    if (auto const it = uv.passenger_groups_.groups_by_source_.find(ds);
        it != end(uv.passenger_groups_.groups_by_source_)) {
      for (auto const pgid : it->second) {
        if (auto const pg = uv.passenger_groups_.at(pgid); pg != nullptr) {
          if (!all_generations && !pg->valid()) {
            continue;
          }
          groups.emplace_back(to_fbs(sched, mc, *pg));
          if (include_localization) {
            localizations.emplace_back(to_fbs_localization_wrapper(
                sched, mc,
                localize(sched,
                         get_reachability(uv, pg->compact_planned_journey_),
                         search_time)));
          }
        }
      }
    }
  };

  for (auto const pgid : *req->ids()) {
    if (auto const pg = uv.passenger_groups_.at(pgid); pg != nullptr) {
      if (all_generations) {
        add_by_data_source(pg->source_);
      } else {
        groups.emplace_back(to_fbs(sched, mc, *pg));
      }
    }
  }

  for (auto const ds : *req->sources()) {
    add_by_data_source(from_fbs(ds));
  }

  mc.create_and_finish(
      MsgContent_PaxMonGetGroupsResponse,
      CreatePaxMonGetGroupsResponse(mc, mc.CreateVector(groups),
                                    mc.CreateVector(localizations))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
