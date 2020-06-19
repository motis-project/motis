#pragma once

#include <ctime>

#include "cista/hash.h"
#include "cista/memory_holder.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"
#include "motis/memory.h"
#include "motis/pair.h"
#include "motis/vector.h"

#include "motis/core/common/fws_multimap.h"
#include "motis/core/schedule/attribute.h"
#include "motis/core/schedule/category.h"
#include "motis/core/schedule/constant_graph.h"
#include "motis/core/schedule/delay_info.h"
#include "motis/core/schedule/event.h"
#include "motis/core/schedule/free_text.h"
#include "motis/core/schedule/nodes.h"
#include "motis/core/schedule/provider.h"
#include "motis/core/schedule/station.h"
#include "motis/core/schedule/trip.h"
#include "motis/core/schedule/waiting_time_rules.h"

namespace motis {

struct schedule {
  schedule() = default;
  schedule(schedule&&) = delete;
  schedule(schedule const&) = delete;
  schedule& operator=(schedule&&) = delete;
  schedule& operator=(schedule const&) = delete;
  ~schedule() = default;

  std::time_t first_event_schedule_time_{std::numeric_limits<time_t>::max()};
  std::time_t last_event_schedule_time_{std::numeric_limits<time_t>::min()};
  std::time_t schedule_begin_{0}, schedule_end_{0};
  mcd::vector<mcd::string> prefixes_;
  mcd::vector<mcd::string> names_;
  cista::hash_t hash_{0U};

  mcd::vector<station_ptr> stations_;
  mcd::hash_map<mcd::string, ptr<station>> eva_to_station_;
  mcd::hash_map<mcd::string, ptr<station>> ds100_to_station_;
  mcd::hash_map<mcd::string, service_class> classes_;
  mcd::vector<mcd::string> tracks_;
  constant_graph travel_time_lower_bounds_fwd_;
  constant_graph travel_time_lower_bounds_bwd_;
  constant_graph transfers_lower_bounds_fwd_;
  constant_graph transfers_lower_bounds_bwd_;
  node_id_t node_count_{0U};
  uint32_t route_count_{0U};
  mcd::vector<station_node_ptr> station_nodes_;
  mcd::vector<ptr<node>> route_index_to_first_route_node_;
  mcd::hash_map<uint32_t, mcd::vector<int32_t>> train_nr_to_routes_;
  waiting_time_rules waiting_time_rules_;

  mcd::vector<mcd::unique_ptr<connection>> full_connections_;
  mcd::vector<mcd::unique_ptr<connection_info>> connection_infos_;
  mcd::vector<mcd::unique_ptr<attribute>> attributes_;
  mcd::vector<mcd::unique_ptr<category>> categories_;
  mcd::vector<mcd::unique_ptr<provider>> providers_;
  mcd::vector<mcd::unique_ptr<mcd::string>> directions_;
  mcd::vector<mcd::unique_ptr<timezone>> timezones_;

  mcd::hash_map<gtfs_trip_id, ptr<trip const>> gtfs_trip_ids_;
  mcd::vector<mcd::pair<primary_trip_id, ptr<trip>>> trips_;
  mcd::vector<mcd::unique_ptr<trip>> trip_mem_;
  mcd::vector<mcd::unique_ptr<mcd::vector<trip::route_edge>>> trip_edges_;
  mcd::vector<mcd::unique_ptr<mcd::vector<ptr<trip>>>> merged_trips_;
  mcd::vector<mcd::unique_ptr<mcd::string>> filenames_;

  std::time_t system_time_{0U}, last_update_timestamp_{0U};
  mcd::vector<mcd::unique_ptr<delay_info>> delay_mem_;
  mcd::hash_map<ev_key, ptr<delay_info>> graph_to_delay_info_;
  mcd::hash_map<ev_key, int32_t> graph_to_track_index_;
  mcd::hash_map<ev_key, mcd::hash_set<free_text>> graph_to_free_texts_;
  mcd::hash_map<ev_key, mcd::vector<ev_key>> waits_for_trains_;
  mcd::hash_map<ev_key, mcd::vector<ev_key>> trains_wait_for_;

  fws_multimap<ptr<trip>> expanded_trips_;
};

using schedule_ptr = mcd::unique_ptr<schedule>;

struct schedule_data {
  schedule_data(cista::memory_holder&& buf, schedule_ptr&& sched)
      : schedule_buf_{std::move(buf)}, schedule_{std::move(sched)} {}
  cista::memory_holder schedule_buf_;
  schedule_ptr schedule_;
};

}  // namespace motis
