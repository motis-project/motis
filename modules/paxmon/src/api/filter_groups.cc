#include "motis/paxmon/api/filter_groups.h"

#include <cstdint>
#include <vector>

#include "utl/verify.h"

#include "motis/hash_set.h"

#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/localization.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/passenger_group.h"
#include "motis/paxmon/reachability.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr filter_groups(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonFilterGroupsRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;
  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  utl::verify(current_time != INVALID_TIME, "invalid current system time");

  auto const only_delayed = req->only_delayed();
  auto const min_delay = req->min_delay();
  auto const only_with_alternative_potential =
      req->only_with_alternative_potential();
  auto const preparation_time = req->preparation_time();
  auto const only_active = req->only_active();
  auto const only_original = req->only_original();
  auto const only_forecast = req->only_forecast();
  auto const include_localization = req->include_localization();

  auto const localization_needed =
      only_with_alternative_potential || include_localization;
  auto const search_time = static_cast<time>(current_time + preparation_time);

  auto total_tracked_groups = 0ULL;
  auto total_active_groups = 0ULL;
  auto filtered_original_groups = 0ULL;
  auto filtered_forecast_groups = 0ULL;
  std::vector<std::uint64_t> selected_group_ids;
  std::vector<passenger_localization> localizations;
  mcd::hash_set<data_source> selected_ds;

  for (auto const pg : uv.passenger_groups_) {
    if (pg == nullptr || (only_active && !pg->valid())) {
      continue;
    }
    ++total_tracked_groups;
    auto const est_arrival = pg->estimated_arrival_time();
    if (est_arrival != INVALID_TIME && est_arrival <= current_time) {
      continue;
    }
    ++total_active_groups;

    if (only_delayed && pg->estimated_delay() < min_delay) {
      continue;
    }

    passenger_localization localization;
    if (localization_needed) {
      auto const reachability =
          get_reachability(uv, pg->compact_planned_journey_);
      localization = localize(sched, reachability, search_time);
      if (only_with_alternative_potential &&
          localization.at_station_->index_ ==
              pg->compact_planned_journey_.destination_station_id()) {
        continue;
      }
    }

    if ((pg->source_flags_ & group_source_flags::FORECAST) ==
        group_source_flags::FORECAST) {
      if (only_original) {
        continue;
      }
      ++filtered_forecast_groups;
    } else {
      if (only_forecast) {
        continue;
      }
      ++filtered_original_groups;
    }

    selected_group_ids.emplace_back(pg->id_);
    selected_ds.insert(pg->source_);
    if (include_localization) {
      localizations.emplace_back(localization);
    }
  }

  message_creator mc;
  mc.create_and_finish(
      MsgContent_PaxMonFilterGroupsResponse,
      CreatePaxMonFilterGroupsResponse(
          mc, total_tracked_groups, total_active_groups,
          selected_group_ids.size(), selected_ds.size(),
          filtered_original_groups, filtered_forecast_groups,
          mc.CreateVector(selected_group_ids),
          mc.CreateVector(utl::to_vec(localizations,
                                      [&](auto const& loc) {
                                        return to_fbs_localization_wrapper(
                                            sched, mc, loc);
                                      })))
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
