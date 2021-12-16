#include "motis/paxmon/track_update.h"

#include "motis/core/schedule/time.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/track_access.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/compact_journey_util.h"
#include "motis/paxmon/util/interchange_time.h"

namespace motis::paxmon {

edge const* find_trip_edge(schedule const& sched, universe const& uv,
                           std::uint32_t station_idx, trip_data_index const tdi,
                           motis::rt::RtEventInfo const* ev) {
  auto const is_dep = ev->event_type() == EventType_DEP;
  auto const schedule_time =
      unix_to_motistime(sched.schedule_begin_, ev->schedule_time());
  for (auto const& ei : uv.trip_data_.edges(tdi)) {
    auto const* trip_edge = ei.get(uv);
    if (is_dep) {
      auto const* dep_node = trip_edge->from(uv);
      if (dep_node->station_idx() == station_idx &&
          dep_node->schedule_time() == schedule_time) {
        return trip_edge;
      }
    } else {
      auto const* arr_node = trip_edge->to(uv);
      if (arr_node->station_idx() == station_idx &&
          arr_node->schedule_time() == schedule_time) {
        return trip_edge;
      }
    }
  }
  return nullptr;
}

void update_track(schedule const& sched, universe& uv,
                  motis::rt::RtTrackUpdate const* tu,
                  std::vector<edge_index>& updated_interchange_edges) {
  ++uv.system_stats_.update_track_count_;
  auto const trp = from_fbs(sched, tu->trip());
  auto const tdi = uv.trip_data_.find_index(trp->trip_idx_);
  if (tdi == INVALID_TRIP_DATA_INDEX) {
    return;
  }
  auto const st = get_station(sched, tu->event()->station_id()->str());
  ++uv.system_stats_.update_track_trip_found_;

  if (st->track_to_platform_.empty()) {
    // no platform information -> no update necessary
    return;
  }

  auto const new_track = get_track_index(sched, tu->updated_track()->str());
  auto const* trip_edge =
      find_trip_edge(sched, uv, st->index_, tdi, tu->event());
  if (trip_edge == nullptr) {
    return;
  }

  if (tu->event()->event_type() == EventType_DEP) {
    auto const* dep_node = trip_edge->from(uv);
    for (auto& ie : dep_node->incoming_edges(uv)) {
      if (ie.is_interchange()) {
        for (auto const& ae : ie.from(uv)->incoming_edges(uv)) {
          if (ae.is_trip()) {
            auto const* arr_node = ae.to(uv);
            if (arr_node->station_idx() != dep_node->station_idx()) {
              continue;
            }
            auto const arr_track = get_arrival_track(
                sched, ae.get_trips(sched).front(), arr_node->station_idx(),
                arr_node->schedule_time());
            auto const new_transfer_time = util::get_interchange_time(
                sched, dep_node->station_idx(), arr_track, new_track);
            if (ie.transfer_time() != new_transfer_time) {
              ie.transfer_time_ = new_transfer_time;
              updated_interchange_edges.emplace_back(get_edge_index(uv, &ie));
            }
          }
        }
      }
    }
  } else {  // arrival
    auto const* arr_node = trip_edge->to(uv);
    for (auto& ie : arr_node->outgoing_edges(uv)) {
      if (ie.is_interchange()) {
        for (auto const& de : ie.from(uv)->outgoing_edges(uv)) {
          if (de.is_trip()) {
            auto const* dep_node = de.to(uv);
            if (dep_node->station_idx() != arr_node->station_idx()) {
              continue;
            }
            auto const dep_track = get_departure_track(
                sched, de.get_trips(sched).front(), dep_node->station_idx(),
                dep_node->schedule_time());
            auto const new_transfer_time = util::get_interchange_time(
                sched, dep_node->station_idx(), new_track, dep_track);
            if (ie.transfer_time() != new_transfer_time) {
              ie.transfer_time_ = new_transfer_time;
              updated_interchange_edges.emplace_back(get_edge_index(uv, &ie));
            }
          }
        }
      }
    }
  }
}

}  // namespace motis::paxmon
