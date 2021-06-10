#pragma once

#include "utl/to_vec.h"

#include "motis/hash_map.h"

#include "motis/core/common/timing.h"
#include "motis/core/schedule/schedule.h"

#include "motis/isochrone/td_dijkstra.h"

namespace motis::isochrone {

struct search_query {
  schedule const* sched_{nullptr};
  node const* from_{nullptr};
  time interval_begin_{0};
  time interval_end_{0};
  bool use_start_footpaths_{false};
  light_connection const* lcon_{nullptr};
};

struct search_result {
  search_result() = default;
  search_result(std::vector<station> stations,
                std::vector<long> travel_times, time interval_begin,
                time interval_end)
      : stations_(std::move(stations)),
        travel_times_(std::move(travel_times)),
        interval_begin_(interval_begin),
        interval_end_(interval_end) {}
  std::vector<station> stations_;
  std::vector<long> travel_times_;
  time interval_begin_{INVALID_TIME};
  time interval_end_{INVALID_TIME};
};

struct search {
  static search_result get_connections(search_query const& q) {

    auto const create_start_edge = [&](node* to) {
      return make_foot_edge(nullptr, to);

    };
    auto mutable_node = const_cast<node*>(q.from_);  // NOLINT
    auto const start_edge = create_start_edge(mutable_node);

    time const schedule_begin = SCHEDULE_OFFSET_MINUTES;
    time const schedule_end =
        (q.sched_->schedule_end_ - q.sched_->schedule_begin_) / 60;


    auto interval_begin = q.interval_begin_;
    auto interval_end = q.interval_end_;




    std::vector<station> stations;
    std::vector<long> travel_times;



    return search_result(stations, travel_times,

                         interval_begin, interval_end);
  }
};

}  // namespace motis::isochrone
