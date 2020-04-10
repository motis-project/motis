#include "motis/csa/collect_start_times.h"

#include "utl/pipes.h"

#include "utl/to_vec.h"

namespace motis::csa {

template <search_dir Dir>
auto station_connections_begin(csa_station const& s) {
  if constexpr (Dir == search_dir::FWD) {
    return s.outgoing_connections_.begin();
  } else {
    return s.incoming_connections_.rbegin();
  }
}

template <search_dir Dir>
auto station_connections_end(csa_station const& s) {
  if constexpr (Dir == search_dir::FWD) {
    return s.outgoing_connections_.end();
  } else {
    return s.incoming_connections_.rend();
  }
}

template <search_dir Dir>
std::set<motis::time> collect_start_times_template(
    csa_timetable const& tt, csa_query const& q, interval const range,
    bool const ontrip_at_interval_end) {
  std::set<motis::time> start_times;
  for (auto const& station_idx : q.meta_starts_) {
    for (auto const& fp : tt.stations_[station_idx].footpaths_) {
      auto const fp_offset =
          fp.from_station_ == fp.to_station_ ? 0U : fp.duration_;
      auto const fp_to_station = tt.stations_[fp.to_station_];
      utl::range{station_connections_begin<Dir>(fp_to_station),
                 station_connections_end<Dir>(fp_to_station)}  //
          | utl::transform([&](csa_connection const* c) {
              return Dir == search_dir::FWD ? c->departure_ - fp_offset
                                            : c->arrival_ + fp_offset;
            })  //
          | utl::unique()  //
          | utl::remove_if([&](motis::time const t) {
              return t < range.begin_ || t > range.end_;
            })  //
          | utl::for_each([&](motis::time const t) { start_times.insert(t); });
    }
  }

  if (ontrip_at_interval_end) {
    start_times.emplace(range.end_ + 1);
  }

  return start_times;
}

std::set<motis::time> collect_start_times(csa_timetable const& tt,
                                          csa_query const& q,
                                          interval const range,
                                          bool const ontrip_at_interval_end) {
  if (q.dir_ == search_dir::FWD) {
    return collect_start_times_template<search_dir::FWD>(
        tt, q, range, ontrip_at_interval_end);
  } else {
    return collect_start_times_template<search_dir::BWD>(
        tt, q, range, ontrip_at_interval_end);
  }
}

}  // namespace motis::csa
