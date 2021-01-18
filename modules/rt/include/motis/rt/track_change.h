#pragma once

#include "utl/verify.h"

#include "motis/core/schedule/build_platform_node.h"
#include "motis/core/schedule/schedule.h"
#include "motis/rt/incoming_edges.h"

namespace motis::rt {

inline void update_platform_edge(schedule& sched, ev_key const& k,
                                 uint16_t old_track, uint16_t new_track) {
  auto const station = sched.stations_[k.get_station_idx()].get();
  auto const old_platform = station->get_platform(old_track);
  auto const new_platform = station->get_platform(new_track);
  if (old_platform == new_platform) {
    return;
  }
  auto const sn = sched.station_nodes_[k.get_station_idx()].get();
  auto rn = k.get_node();
  std::vector<incoming_edge_patch> incoming;
  save_outgoing_edges(std::set<station_node*>{sn}, incoming);
  save_outgoing_edges(rn, incoming);
  if (old_platform) {
    // disable old edge
    auto old_pn = sn->platform_nodes_.at(old_platform.value());
    utl::verify(old_pn != nullptr, "invalid old platform");
    if (k.is_departure()) {
      for (auto& e : old_pn->edges_) {
        if (e.to_ == rn) {
          e.m_.type_ = edge::INVALID_EDGE;
        }
      }
    } else {
      for (auto& e : rn->edges_) {
        if (e.to_ == old_pn) {
          e.m_.type_ = edge::INVALID_EDGE;
        }
      }
    }
  }
  if (new_platform) {
    // add new edge
    if (k.is_departure()) {
      auto const pn = add_platform_enter_edge(sched, rn, sn,
                                              station->platform_transfer_time_,
                                              new_platform.value());
      add_outgoing_edge(&pn->edges_.back(), incoming);
    } else {
      add_platform_exit_edge(sched, rn, sn, station->platform_transfer_time_,
                             new_platform.value());
      add_outgoing_edge(&rn->edges_.back(), incoming);
    }
  }
  patch_incoming_edges(incoming);
}

inline void update_track(schedule& sched, ev_key const& k, uint16_t new_track) {
  auto fcon = *k.lcon()->full_con_;
  auto& track = (k.ev_type_ == event_type::ARR ? fcon.a_track_ : fcon.d_track_);
  auto const old_track = track;
  track = new_track;

  const_cast<light_connection*>(k.lcon())->full_con_ =  // NOLINT
      sched.full_connections_.emplace_back(mcd::make_unique<connection>(fcon))
          .get();

  update_platform_edge(sched, k, old_track, new_track);
}

}  // namespace motis::rt
