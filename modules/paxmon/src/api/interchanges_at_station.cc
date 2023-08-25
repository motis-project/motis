#include "motis/paxmon/api/interchanges_at_station.h"

#include <algorithm>
#include <utility>

#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/conv/station_conv.h"

#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/util/interchange_info.h"

using namespace motis::module;
using namespace motis::paxmon;
using namespace motis::paxmon::util;

namespace motis::paxmon::api {

msg_ptr interchanges_at_station(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonInterchangesAtStationRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto const& uv = uv_access.uv_;
  auto const& ic_station = *get_station(sched, req->station()->str());

  // filters
  auto const start_time =
      req->filter_interval()->begin() != 0
          ? unix_to_motistime(sched.schedule_begin_,
                              req->filter_interval()->begin())
          : 0;
  auto const end_time = req->filter_interval()->end() != 0
                            ? unix_to_motistime(sched.schedule_begin_,
                                                req->filter_interval()->end())
                            : std::numeric_limits<time>::max();

  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  auto const ignore_past_transfers =
      req->ignore_past_transfers() && current_time != INVALID_TIME;

  auto const filter_times = start_time != 0 ||
                            end_time != std::numeric_limits<time>::max() ||
                            ignore_past_transfers;
  auto const include_group_infos = req->include_group_infos();
  auto const include_broken_interchanges = req->include_broken_interchanges();
  auto const include_disabled_group_routes =
      req->include_disabled_group_routes();

  auto const max_results = req->max_results();

  message_creator mc;
  auto interchange_infos =
      std::vector<flatbuffers::Offset<PaxMonInterchangeInfo>>{};
  auto max_count_reached = false;

  auto const include_event = [&](event_node const* ev) {
    if (!filter_times) {
      return true;
    }
    if (ev->is_enter_exit_node()) {
      return false;
    }
    if (ignore_past_transfers && ev->current_time() < current_time) {
      return false;
    }
    return (ev->schedule_time() >= start_time &&
            ev->schedule_time() <= end_time) ||
           (ev->current_time() >= start_time && ev->current_time() <= end_time);
  };

  auto const gii_options = get_interchange_info_options{
      .include_group_infos_ = include_group_infos,
      .include_disabled_group_routes_ = include_disabled_group_routes,
  };

  auto visited_stations = std::set<unsigned>{};
  auto const add_station = [&](unsigned const station_idx) {
    if (!visited_stations.insert(station_idx).second) {
      return;
    }
    if (station_idx >= uv.interchanges_at_station_.index_size()) {
      return;
    }
    for (auto const& ei : uv.interchanges_at_station_.at(station_idx)) {
      auto const* ic_edge = ei.get(uv);
      if ((!include_broken_interchanges && !ic_edge->is_valid(uv)) ||
          (!include_event(ic_edge->from(uv)) &&
           !include_event(ic_edge->to(uv)))) {
        continue;
      }

      interchange_infos.emplace_back(
          get_interchange_info(uv, sched, ei, mc, gii_options)
              .to_fbs_interchange_info(mc, uv, sched, false));

      if (max_results != 0 && interchange_infos.size() >= max_results) {
        max_count_reached = true;
        break;
      }
    }
  };

  add_station(ic_station.index_);
  if (req->include_meta_stations()) {
    for (auto const& eq_station : ic_station.equivalent_) {
      add_station(eq_station->index_);
    }
  }

  mc.create_and_finish(
      MsgContent_PaxMonInterchangesAtStationResponse,
      CreatePaxMonInterchangesAtStationResponse(
          mc, to_fbs(mc, ic_station), mc.CreateVector(interchange_infos),
          max_count_reached)
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
