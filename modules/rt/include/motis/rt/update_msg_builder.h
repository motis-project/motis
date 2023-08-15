#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ctx/res_id_t.h"

#include "motis/core/schedule/delay_info.h"
#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"
#include "motis/hash_map.h"
#include "motis/hash_set.h"

#include "motis/rt/expanded_trips.h"

namespace motis::rt {

struct expanded_trip_update_info {
  std::optional<expanded_trip_index> old_route_{};
  std::optional<expanded_trip_index> new_route_{};
};

struct rt_update_info {
  Content content_type_{Content_NONE};
  flatbuffers::Offset<void> content_{0};
  bool intermediate_{false};
};

struct update_msg_builder {
  update_msg_builder(schedule const& sched, ctx::res_id_t schedule_res_id);

  void add_delay(delay_info const* di);

  void trip_separated(trip const* trp);

  void add_reroute(trip const* trp,
                   mcd::vector<trip::route_edge> const& old_edges,
                   lcon_idx_t old_lcon_idx);

  void add_free_text_nodes(trip const* trp, free_text const& ft,
                           std::vector<ev_key> const& events);

  void add_track_nodes(ev_key const& k, std::string const& track,
                       motis::time schedule_time);

  void add_station(node_id_t station_idx);

  void expanded_trip_added(trip const* trp, expanded_trip_index eti);

  void expanded_trip_moved(trip const* trp,
                           std::optional<expanded_trip_index> old_eti,
                           std::optional<expanded_trip_index> new_eti);

  void trip_formation_message(motis::ris::TripFormationMessage const* msg);

  motis::module::msg_ptr finish();

  bool should_finish() const;

  std::size_t delay_count() const { return delay_count_; }
  std::size_t reroute_count() const { return reroute_count_; }

  void reset();

private:
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<RtEventInfo>>>
  to_fbs_event_infos(mcd::vector<trip::route_edge> const& edges,
                     lcon_idx_t lcon_idx);

  void build_delay_updates();
  void build_expanded_trip_updates();

  std::pair<expanded_trip_update_info&, bool /*  inserted */>
  get_or_insert_expanded_trip(trip const* trp);

  motis::module::message_creator fbb_;
  schedule const& sched_;
  ctx::res_id_t schedule_res_id_;

  std::vector<rt_update_info> updates_;
  mcd::hash_map<trip const*, std::vector<delay_info const*>> delays_;
  mcd::hash_map<trip const*, expanded_trip_update_info> expanded_trips_;
  mcd::hash_set<trip const*> separated_trips_;
  mcd::hash_map<trip const*, std::size_t> previous_reroute_update_;
  mcd::hash_map<std::string, std::size_t> previous_trip_formation_update_;
  mcd::hash_map<ev_key, std::size_t> previous_track_update_;
  std::size_t delay_count_{0};
  std::size_t reroute_count_{0};
};

}  // namespace motis::rt
