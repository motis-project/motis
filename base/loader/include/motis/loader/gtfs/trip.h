#pragma once

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "cista/reflection/comparable.h"

#include "motis/loader/gtfs/flat_map.h"
#include "motis/loader/gtfs/route.h"
#include "motis/loader/gtfs/services.h"
#include "motis/loader/gtfs/stop.h"
#include "motis/loader/loaded_file.h"

namespace motis::loader::gtfs {

struct trip;

struct block {
  std::vector<std::pair<std::vector<trip*>, bitfield>> rule_services();
  std::vector<trip*> trips_;
};

using block_map = std::map<std::string, std::unique_ptr<block>>;

struct stop_time {
  stop_time();
  stop_time(stop*, std::string headsign, int arr_time, bool out_allowed,
            int dep_time, bool in_allowed);

  struct ev {
    int time_{0};
    bool in_out_allowed_{false};
  };

  stop* stop_{nullptr};
  std::string headsign_;
  ev arr_, dep_;
};

struct trip {
  struct stop_identity {
    CISTA_COMPARABLE()
    stop* stop_{nullptr};
    bool out_allowed_{false};
    bool in_allowed_{false};
  };
  using stop_seq = std::vector<stop_identity>;

  trip(route const*, bitfield const*, block*, std::string id,
       std::string headsign, std::string short_name, unsigned line);

  stop_seq stops() const;

  int avg_speed() const;
  int distance() const;

  route const* route_;
  bitfield const* service_;
  block* block_;
  std::string id_;
  std::string headsign_;
  std::string short_name_;
  flat_map<stop_time> stop_times_;
  unsigned line_;
};

using trip_map = std::map<std::string, std::unique_ptr<trip>>;

std::pair<trip_map, block_map> read_trips(loaded_file, route_map const&,
                                          traffic_days const&);

}  // namespace motis::loader::gtfs
