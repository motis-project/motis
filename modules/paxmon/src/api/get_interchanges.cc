#include "motis/paxmon/api/get_interchanges.h"

#include <algorithm>
#include <utility>

#include "motis/core/access/station_access.h"
#include "motis/core/access/time_access.h"
#include "motis/core/conv/station_conv.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/get_universe.h"
#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace motis::paxmon;

namespace motis::paxmon::api {

msg_ptr get_interchanges(paxmon_data& data, msg_ptr const& msg) {
  auto const req = motis_content(PaxMonGetInterchangesRequest, msg);
  auto const uv_access = get_universe_and_schedule(data, req->universe());
  auto const& sched = uv_access.sched_;
  auto& uv = uv_access.uv_;
  auto const& ic_station = *get_station(sched, req->station()->str());

  // filters
  auto const start_time =
      req->start_time() != 0 ? unix_to_motistime(sched, req->start_time()) : 0;
  auto const end_time = req->end_time() != 0
                            ? unix_to_motistime(sched, req->end_time())
                            : std::numeric_limits<time>::max();
  auto const filter_times = req->start_time() != 0 || req->end_time() != 0;
  auto const max_count = req->max_count();
  auto const include_group_infos = req->include_group_infos();
  auto const include_broken_interchanges = req->include_broken_interchanges();
  auto const include_disabled_group_routes =
      req->include_disabled_group_routes();

  message_creator mc;
  std::vector<flatbuffers::Offset<PaxMonInterchangeInfo>> interchange_infos;
  auto max_count_reached = false;

  auto const include_event = [&](event_node const* ev) {
    if (!filter_times) {
      return true;
    }
    if (ev->is_enter_exit_node()) {
      return false;
    }
    return (ev->schedule_time() >= start_time &&
            ev->schedule_time() <= end_time) ||
           (ev->current_time() >= start_time && ev->current_time() <= end_time);
  };

  auto const make_fbs_event = [&](event_node const* ev, bool const arrival) {
    std::vector<flatbuffers::Offset<PaxMonTripStopInfo>> res;
    if (!ev->is_enter_exit_node()) {
      std::vector<flatbuffers::Offset<TripServiceInfo>> fbs_trips;
      // TODO(pablo): service infos only for arriving trip section
      if (arrival) {
        for (auto const& trp_edge : ev->incoming_edges(uv)) {
          if (trp_edge.is_trip()) {
            for (auto const& trp : trp_edge.get_trips(sched)) {
              fbs_trips.emplace_back(to_fbs_trip_service_info(mc, sched, trp));
            }
          }
        }
      } else {
        for (auto const& trp_edge : ev->outgoing_edges(uv)) {
          if (trp_edge.is_trip()) {
            for (auto const& trp : trp_edge.get_trips(sched)) {
              fbs_trips.emplace_back(to_fbs_trip_service_info(mc, sched, trp));
            }
          }
        }
      }
      res.emplace_back(CreatePaxMonTripStopInfo(
          mc, motis_to_unixtime(sched, ev->schedule_time()),
          motis_to_unixtime(sched, ev->current_time()),
          mc.CreateVector(fbs_trips),
          to_fbs(mc, *sched.stations_.at(ev->station_idx()))));
    }
    return mc.CreateVector(res);
  };

  std::set<unsigned> visited_stations;
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

      std::vector<PaxMonGroupRouteBaseInfo> group_route_infos;
      if (include_group_infos) {
        for (auto const& pgwr :
             uv.pax_connection_info_.group_routes(ic_edge->pci_)) {
          auto const& gr = uv.passenger_groups_.route(pgwr);
          if (include_disabled_group_routes || gr.probability_ != 0.0F) {
            auto const& pg = uv.passenger_groups_.group(pgwr.pg_);
            group_route_infos.emplace_back(to_fbs_base_info(mc, pg, gr));
          }
        }
        std::sort(begin(group_route_infos), end(group_route_infos),
                  [](PaxMonGroupRouteBaseInfo const& a,
                     PaxMonGroupRouteBaseInfo const& b) {
                    return std::make_pair(a.g(), a.r()) <
                           std::make_pair(b.g(), b.r());
                  });
      }
      auto const pdf =
          get_load_pdf(uv.passenger_groups_,
                       uv.pax_connection_info_.group_routes(ic_edge->pci_));
      auto const cdf = get_cdf(pdf);

      interchange_infos.emplace_back(CreatePaxMonInterchangeInfo(
          mc, make_fbs_event(ic_edge->from(uv), true),
          make_fbs_event(ic_edge->to(uv), false),
          CreatePaxMonCombinedGroupRoutes(
              mc, mc.CreateVectorOfStructs(group_route_infos),
              to_fbs_distribution(mc, pdf, cdf)),
          ic_edge->transfer_time(), ic_edge->is_valid(uv),
          ic_edge->is_disabled(), ic_edge->broken_));

      if (max_count != 0 && interchange_infos.size() >= max_count) {
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
      MsgContent_PaxMonGetInterchangesResponse,
      CreatePaxMonGetInterchangesResponse(mc, to_fbs(mc, ic_station),
                                          mc.CreateVector(interchange_infos),
                                          max_count_reached)
          .Union());
  return make_msg(mc);
}

}  // namespace motis::paxmon::api
