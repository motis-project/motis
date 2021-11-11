#pragma once

#include <iostream>

#include "utl/enumerate.h"
#include "utl/to_vec.h"

#include "motis/raptor/raptor_result.h"
#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

inline std::string get_string(stop_id const s_id,
                              raptor_meta_info const& sched) {
  return "SID: " + std::to_string(s_id) +
         " -- EVA: " + sched.raptor_id_to_eva_[s_id];
}

inline void print_station(stop_id const s_id, raptor_meta_info const& sched) {
  std::cout << "SID: " << s_id << " -- EVA: " << sched.raptor_id_to_eva_[s_id]
            << '\n';
}

template <typename TimeStringer>
inline void print_route_gen(route_id const r_id, raptor_timetable const& tt,
                            TimeStringer const& time_string) {
  auto const& route = tt.routes_[r_id];

  auto stop_count = route.stop_count_;
  auto index_into_route_stops = route.index_to_route_stops_;
  auto index_into_stop_times = route.index_to_stop_times_;

  std::cout << r_id << "\t{ ";
  for (stop_id stop_offset = 0; stop_offset < stop_count; ++stop_offset) {
    std::cout << stop_offset << ": "
              << tt.route_stops_[index_into_route_stops + stop_offset] << " ";
  }
  std::cout << "} " << stop_count << "\n";

  for (trip_count trip_offset = 0; trip_offset < route.trip_count_;
       ++trip_offset) {
    std::cout << trip_offset << " \t[ ";
    for (stop_id stop_offset = 0; stop_offset < stop_count; ++stop_offset) {
      auto const st_idx =
          index_into_stop_times + (trip_offset * stop_count) + stop_offset;
      auto const stop_time = tt.stop_times_[st_idx];
      std::cout << stop_offset << ": "
                << "(" << time_string(stop_time.arrival_) << ","
                << time_string(stop_time.departure_) << ") ; ";
    }
    std::cout << "]\n";
  }
}

inline void print_route(route_id const r_id, raptor_timetable const& tt) {
  print_route_gen(r_id, tt, [](time const t) { return std::to_string(t); });
}

inline void print_route_format(route_id const r_id,
                               raptor_timetable const& tt) {
  print_route_gen(r_id, tt, [](time const t) { return format_time(t); });
}

template <typename Container>
inline void print_routes(Container const& r_ids, raptor_timetable const& tt) {
  for (auto const& r_id : r_ids) {
    print_route(r_id, tt);
  }
}

inline void print_footpaths(raptor_timetable const& tt) {
  std::cout << "Printing all footpaths\n";
  for (auto const& [s_idx, s] : utl::enumerate(tt.stops_)) {
    for (auto f_idx = s.index_to_transfers_;
         f_idx < s.index_to_transfers_ + s.footpath_count_; ++f_idx) {
      auto const& fp = tt.footpaths_[f_idx];
      std::cout << "FP [ " << s_idx << " -> " << fp.to_ << " : "
                << static_cast<motis::time>(fp.duration_) << " ]\n";
    }
  }
}

template <typename Container>
inline void print_routes_format(Container const& r_ids,
                                raptor_timetable const& tt) {
  for (auto const& r_id : r_ids) {
    print_route_format(r_id, tt);
  }
}

inline void print_station_arrivals(stop_id const s_id,
                                   raptor_result_base const& raptor_result) {
  std::cout << s_id << "(station) Arrivals: [ ";
  for (auto k = 0; k < max_raptor_round; ++k) {
    std::cout << raptor_result[k][s_id] << " ";
  }
  std::cout << "]\n";
}

inline std::vector<route_id> routes_from_station(stop_id const s_id,
                                                 raptor_timetable const& tt) {
  std::vector<route_id> routes;

  auto const& station = tt.stops_[s_id];
  auto const& next_station = tt.stops_[s_id + 1];
  for (auto stop_routes_idx = station.index_to_stop_routes_;
       stop_routes_idx < next_station.index_to_stop_routes_;
       ++stop_routes_idx) {
    routes.push_back(tt.stop_routes_[stop_routes_idx]);
  }

  return routes;
}

template <typename Container>
inline std::vector<route_id> get_routes_containing(Container const& stations,
                                                   raptor_timetable const& tt) {
  std::vector<route_id> routes;

  for (route_id r_id = 0; r_id < tt.route_count(); ++r_id) {
    auto const& route = tt.routes_[r_id];

    auto const route_begin =
        std::begin(tt.route_stops_) + route.index_to_route_stops_;
    auto const route_end = route_begin + route.stop_count_;

    bool contains_all = true;
    for (auto const s_id : stations) {
      auto found = std::find(route_begin, route_end, s_id);
      if (found == route_end) {
        contains_all = false;
        break;
      }
    }

    if (contains_all) {
      routes.emplace_back(r_id);
    }
  }

  return routes;
}

template <typename Container>
inline std::vector<route_id> get_routes_containing_evas(
    Container const& evas, raptor_meta_info const& raptor_sched,
    raptor_timetable const& tt) {
  return get_routes_containing(
      utl::to_vec(
          evas,
          [&](auto&& eva) { return raptor_sched.eva_to_raptor_id_.at(eva); }),
      tt);
}

inline void print_route_arrivals(route_id const r_id,
                                 raptor_timetable const& tt,
                                 time const* const arrs) {
  auto const& route = tt.routes_[r_id];
  auto const base_rsi = route.index_to_route_stops_;
  std::cout << "[ ";
  for (stop_offset so = 0; so < route.stop_count_; ++so) {
    auto const rs = tt.route_stops_[base_rsi + so];
    std::cout << rs << ":" << arrs[rs] << " ";
  }
  std::cout << "]\n";
}

inline bool is_reset(raptor_result_base const& result) {
  for (auto k = 0; k < max_raptor_round; ++k) {
    for (auto s = 0; s < result.stop_count_; ++s) {
      if (result[k][s] != invalid<time>) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace motis::raptor