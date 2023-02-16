#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "cista/reflection/comparable.h"

#include "motis/loader/gtfs/flat_map.h"
#include "motis/loader/gtfs/parse_time.h"
#include "motis/loader/gtfs/route.h"
#include "motis/loader/gtfs/services.h"
#include "motis/loader/gtfs/stop.h"
#include "motis/loader/loaded_file.h"
#include "motis/schedule-format/Service_generated.h"

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
    int time_{kInterpolate};
    bool in_out_allowed_{false};
  };

  stop* stop_{nullptr};
  std::string headsign_;
  ev arr_, dep_;
};

struct frequency {
  int start_time_;  // minutes since midnight
  int end_time_;  // minutes since midnight on start day
  int headway_;  // minutes between trip starts
  ScheduleRelationship schedule_relationship_;
};

struct trip {
  struct stop_identity {
    CISTA_COMPARABLE()
    stop* stop_{nullptr};
    bool out_allowed_{false};
    bool in_allowed_{false};
  };
  using stop_seq = std::vector<stop_identity>;
  using stop_seq_numbers = std::vector<unsigned>;

  trip(route const*, bitfield const*, block*, std::string id,
       std::string headsign, std::string short_name, std::size_t line);

  void interpolate();

  stop_seq stops() const;
  stop_seq_numbers seq_numbers() const;

  int avg_speed() const;
  int distance() const;

  void expand_frequencies(
      std::function<void(trip const&, ScheduleRelationship)> const&) const;

  void print_stop_times(std::ostream&, unsigned indent = 0) const;

  route const* route_;
  bitfield const* service_;
  block* block_;
  std::string id_;
  std::string headsign_;
  std::string short_name_;
  flat_map<stop_time> stop_times_;
  std::size_t line_;
  std::optional<std::vector<frequency>> frequency_;
};

using trip_map = std::map<std::string, std::unique_ptr<trip>>;

std::pair<trip_map, block_map> read_trips(loaded_file, route_map const&,
                                          traffic_days const&);

void read_frequencies(loaded_file, trip_map&);

}  // namespace motis::loader::gtfs
