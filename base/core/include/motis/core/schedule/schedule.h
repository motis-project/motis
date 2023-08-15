#pragma once

#include <ctime>

#include "boost/uuid/uuid.hpp"

#include "cista/hash.h"
#include "cista/memory_holder.h"

#include "motis/hash_map.h"
#include "motis/hash_set.h"
#include "motis/memory.h"
#include "motis/pair.h"
#include "motis/vector.h"

#include "motis/core/common/dynamic_fws_multimap.h"
#include "motis/core/common/fws_multimap.h"
#include "motis/core/common/unixtime.h"
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
  unixtime first_event_schedule_time_{std::numeric_limits<time_t>::max()};
  unixtime last_event_schedule_time_{std::numeric_limits<time_t>::min()};
  unixtime schedule_begin_{0}, schedule_end_{0};
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
  node_id_t next_node_id_{0U};
  node_id_t non_station_node_offset_{1'000'000U};
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
  mcd::hash_map<mcd::string, ptr<provider>> provider_by_full_name_;
  mcd::vector<mcd::unique_ptr<mcd::string>> directions_;
  mcd::vector<mcd::unique_ptr<timezone>> timezones_;

  mcd::hash_map<
      mcd::string,
      mcd::vector<mcd::pair<unixtime /* trip start time */, ptr<trip const>>>>
      gtfs_trip_ids_;
  mcd::vector<mcd::pair<primary_trip_id, ptr<trip>>> trips_;
  mcd::vector<mcd::unique_ptr<trip>> trip_mem_;
  mcd::vector<mcd::unique_ptr<mcd::vector<trip::route_edge>>> trip_edges_;
  mcd::vector<mcd::unique_ptr<mcd::vector<ptr<trip>>>> merged_trips_;
  mcd::vector<mcd::unique_ptr<mcd::string>> filenames_;
  mcd::hash_map<boost::uuids::uuid, ptr<trip>> uuid_to_trip_;

  unixtime system_time_{0U}, last_update_timestamp_{0U};
  mcd::vector<mcd::unique_ptr<delay_info>> delay_mem_;
  mcd::hash_map<ev_key, ptr<delay_info>> graph_to_delay_info_;
  mcd::hash_map<ev_key, uint16_t> graph_to_schedule_track_index_;
  mcd::hash_map<ev_key, mcd::hash_set<free_text>> graph_to_free_texts_;
  mcd::hash_map<ev_key, mcd::vector<ev_key>> waits_for_trains_;
  mcd::hash_map<ev_key, mcd::vector<ev_key>> trains_wait_for_;

  mcd::hash_map<boost::uuids::uuid, mcd::pair<ptr<trip const>, ev_key>>
      uuid_to_event_;
  mcd::hash_map<mcd::pair<ptr<trip const>, ev_key>, boost::uuids::uuid>
      event_to_uuid_;

  dynamic_fws_multimap<ptr<trip>> expanded_trips_;
  dynamic_fws_multimap<uint32_t> route_to_expanded_routes_;
};

using schedule_ptr = mcd::unique_ptr<schedule>;

struct schedule_data {
  schedule_data(cista::memory_holder&& buf, schedule_ptr&& sched)
      : schedule_buf_{std::move(buf)}, schedule_{std::move(sched)} {}
  cista::memory_holder schedule_buf_;
  schedule_ptr schedule_;
};

}  // namespace motis
