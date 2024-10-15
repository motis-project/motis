#pragma once

#include <vector>

#include "ctx/res_id_t.h"

#include "motis/core/schedule/delay_info.h"
#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"
#include "motis/hash_map.h"

namespace motis::rt {

struct update_msg_builder {
  update_msg_builder(schedule const& sched, ctx::res_id_t schedule_res_id);

  void add_delay(delay_info const* di);

  void add_reroute(trip const* trp,
                   mcd::vector<trip::route_edge> const& old_edges,
                   lcon_idx_t old_lcon_idx);

  void add_free_text_nodes(trip const* trp, free_text const& ft,
                           std::vector<ev_key> const& events);

  void add_track_nodes(ev_key const& k, std::string const& track,
                       motis::time schedule_time);

  void add_station(node_id_t station_idx);

  motis::module::msg_ptr finish();

  std::size_t delay_count() const { return delay_count_; }
  std::size_t reroute_count() const { return reroute_count_; }

  void reset();

private:
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<RtEventInfo>>>
  to_fbs_event_infos(mcd::vector<trip::route_edge> const& edges,
                     lcon_idx_t lcon_idx);

  void build_delay_updates();

  motis::module::message_creator fbb_;
  schedule const& sched_;
  ctx::res_id_t schedule_res_id_;
  std::vector<flatbuffers::Offset<RtUpdate>> updates_;
  mcd::hash_map<trip const*, std::vector<delay_info const*>> delays_;
  std::size_t delay_count_{0};
  std::size_t reroute_count_{0};
};

}  // namespace motis::rt
