#include "motis/csa/build_csa_timetable.h"

#include <ciso646>
#include <algorithm>
#include <functional>
#include <map>
#include <queue>
#include <set>
#include <thread>

#include "boost/sort/sort.hpp"

#include "utl/pipes/range.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/edges.h"
#include "motis/core/schedule/nodes.h"
#include "motis/core/schedule/station.h"
#include "motis/core/schedule/time.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_iterator.h"

using namespace motis::logging;
using namespace motis::access;

namespace motis::csa {

namespace {

constexpr auto const BUCKET_SIZE = 1U;
inline int get_bucket(time const t) { return t / BUCKET_SIZE; }
inline bool is_same_bucket(time const a, time const b) {
  return get_bucket(a) == get_bucket(b);
};

void init_trip_to_connections(csa_timetable& tt) {
  scoped_timer timer("csa: trip to connections");
  tt.trip_to_connections_.resize(tt.trip_count_);
  for (auto const& con : tt.fwd_connections_) {
    if (con.light_con_ != nullptr) {
      assert(tt.trip_to_connections_[con.trip_].size() == con.trip_con_idx_);
      tt.trip_to_connections_[con.trip_].emplace_back(&con);
    }
  }
}

void init_stop_to_connections(csa_timetable& tt) {
  scoped_timer timer("csa: stop to connections");
  for (auto const& con : tt.fwd_connections_) {
    if (con.light_con_ != nullptr) {
      tt.stations_[con.from_station_].outgoing_connections_.emplace_back(&con);
      tt.stations_[con.to_station_].incoming_connections_.emplace_back(&con);
    }
  }
}

std::vector<uint32_t> get_bucket_starts(
    std::vector<csa_connection>::const_iterator const it_begin,
    std::vector<csa_connection>::const_iterator const it_end,
    search_dir const dir, bool const bridged) {
  if (it_begin == it_end) {
    return {};
  }

  auto const get_start_bucket = [&](csa_connection const& c) {
    return get_bucket(dir == search_dir::FWD ? c.departure_ : c.arrival_);
  };
  auto const get_dest_bucket = [&](csa_connection const& c) {
    return get_bucket(dir == search_dir::FWD ? c.arrival_ : c.departure_);
  };
  auto const get_start_station = [&](csa_connection const& c) {
    return dir == search_dir::FWD ? c.from_station_ : c.to_station_;
  };
  auto const get_dest_station = [&](csa_connection const& c) {
    return dir == search_dir::FWD ? c.to_station_ : c.from_station_;
  };

  auto bucket_starts = std::vector<uint32_t>{0U};
  auto curr_bucket = get_start_bucket(*it_begin);

  if (it_begin != it_end) {
    auto i = 1U;

    if (bridged) {
      // Bridge connections were inserted.
      // Buckets = minutes.
      for (auto const& curr_con : utl::range{std::next(it_begin), it_end}) {
        if (curr_bucket != get_start_bucket(curr_con)) {
          bucket_starts.emplace_back(i);
          curr_bucket = get_start_bucket(curr_con);
        }
        ++i;
      }
    } else {
      // Stores arrivals of connections
      // that depart and arrive within the same bucket.
      auto same_bucket_arrival =
          std::set<std::pair<station_id /* arrival station */,
                             time /* arrival time */>>{};
      auto curr_start_bucket = get_start_bucket(*it_begin);
      for (auto it = std::next(it_begin); it != it_end; ++it, ++i) {
        auto const& curr_con = *it;
        if (curr_start_bucket != get_start_bucket(curr_con) ||
            same_bucket_arrival.find(
                {get_start_station(curr_con), get_start_bucket(curr_con)}) !=
                end(same_bucket_arrival)) {
          bucket_starts.emplace_back(i);
          same_bucket_arrival.clear();
          curr_start_bucket = get_start_bucket(curr_con);
        }
        if (curr_con.get_duration() == 0U) {
          same_bucket_arrival.emplace(get_dest_station(curr_con),
                                      get_dest_bucket(curr_con));
        }
      }
    }
  }

  bucket_starts.emplace_back(
      static_cast<uint32_t>(std::distance(it_begin, it_end)));

  LOG(info) << "CSA bucket count: " << bucket_starts.size() - 1;

  return bucket_starts;
}

trip_id get_connections_from_expanded_trips(
    csa_timetable& tt, schedule const& sched,
    bool bridge_zero_duration_connections, bool add_footpath_connections) {
  scoped_timer build_timer{"csa: get connections"};
  trip_id trip_idx = 0;
  auto bridged_count = 0U;
  auto bridged_footpath_count = 0U;
  auto footpath_connections_count = 0U;
  {
    scoped_timer connections_timer{"csa: build connections"};
    for (auto const& route_trips : sched.expanded_trips_) {
      utl::verify(!route_trips.empty(), "empty route");
      auto const first_trip = route_trips[0];
      auto const in_allowed =
          utl::to_vec(stops(first_trip), [](trip_stop const& ts) {
            return ts.get_route_node()->is_in_allowed();
          });
      auto const out_allowed =
          utl::to_vec(stops(first_trip), [](trip_stop const& ts) {
            return ts.get_route_node()->is_out_allowed();
          });

      for (auto const& trp : route_trips) {
        auto const trp_sections = sections{trp};
        for (auto sec_it = trp_sections.begin(); sec_it != trp_sections.end();
             ++sec_it) {
          auto const& s = *sec_it;
          auto const& lc = s.lcon();

          if (bridge_zero_duration_connections) {
            auto price_sum = s.fcon().price_;
            for (auto const& following_sec :
                 utl::range{std::next(sec_it), end(trp_sections)}) {
              if (!is_same_bucket(lc.d_time_, following_sec.lcon().d_time_)) {
                break;
              }
              price_sum += following_sec.fcon().price_;
              tt.fwd_connections_.emplace_back(
                  s.from_station_id(), following_sec.to_station_id(),
                  lc.d_time_, following_sec.lcon().a_time_, price_sum, trip_idx,
                  static_cast<con_idx_t>(s.index()), in_allowed[s.index()],
                  out_allowed[following_sec.index() + 1], lc.full_con_->clasz_,
                  nullptr);
              ++bridged_count;

              if (add_footpath_connections) {
                for (auto const& fp :
                     tt.stations_[following_sec.to_station_id()].footpaths_) {
                  if (fp.from_station_ != fp.to_station_) {
                    tt.fwd_connections_.emplace_back(
                        s.from_station_id(), fp.to_station_, lc.d_time_,
                        following_sec.lcon().a_time_ + fp.duration_ -
                            tt.stations_[fp.to_station_].transfer_time_,
                        price_sum, trip_idx, s.index(), in_allowed[s.index()],
                        out_allowed[following_sec.index() + 1], s.fcon().clasz_,
                        nullptr);
                  }
                  ++bridged_footpath_count;
                  ++footpath_connections_count;
                }
              }
            }
          }

          if (add_footpath_connections) {
            for (auto const& fp : tt.stations_[s.to_station_id()].footpaths_) {
              if (fp.from_station_ != fp.to_station_) {
                tt.fwd_connections_.emplace_back(
                    s.from_station_id(), fp.to_station_, lc.d_time_,
                    lc.a_time_ + fp.duration_ -
                        tt.stations_[fp.to_station_].transfer_time_,
                    s.fcon().price_, trip_idx, s.index(), in_allowed[s.index()],
                    out_allowed[s.index() + 1], s.fcon().clasz_, nullptr);
                ++footpath_connections_count;
              }
            }
          }

          auto const from = s.from_station_id();
          auto const to = s.to_station_id();
          auto const from_in_allowed = in_allowed[s.index()];
          auto const to_out_allowed = out_allowed[s.index() + 1];
          tt.fwd_connections_.emplace_back(
              from, to, lc.d_time_, lc.a_time_, s.fcon().price_, trip_idx,
              static_cast<con_idx_t>(s.index()), from_in_allowed,
              to_out_allowed, lc.full_con_->clasz_, &lc);
        }
        ++trip_idx;
      }
    }
    LOG(info) << "CSA added bridge connections (bucket size " << BUCKET_SIZE
              << "): " << bridged_count;
    LOG(info) << "CSA added footpath connections: "
              << footpath_connections_count;
    LOG(info) << "CSA bridged footpath connections: " << bridged_footpath_count;
  }

  {
    scoped_timer sort_timer{"csa: sort connections"};
    boost::sort::parallel_stable_sort(
        std::begin(tt.fwd_connections_), std::end(tt.fwd_connections_),
        [&](csa_connection const& c1, csa_connection const& c2) -> bool {
          return c1.departure_ < c2.departure_;
        });

    tt.bwd_connections_ = tt.fwd_connections_;
    std::reverse(std::begin(tt.bwd_connections_),
                 std::end(tt.bwd_connections_));
    boost::sort::parallel_stable_sort(
        std::begin(tt.bwd_connections_), std::end(tt.bwd_connections_),
        [&](csa_connection const& c1, csa_connection const& c2) -> bool {
          return c1.arrival_ > c2.arrival_;
        });
  }

  {
    scoped_timer sort_timer{"csa: compute buckets"};
    tt.fwd_bucket_starts_ =
        get_bucket_starts(begin(tt.fwd_connections_), end(tt.fwd_connections_),
                          search_dir::FWD, bridge_zero_duration_connections);
    tt.bwd_bucket_starts_ =
        get_bucket_starts(begin(tt.bwd_connections_), end(tt.bwd_connections_),
                          search_dir::BWD, bridge_zero_duration_connections);
  }

  assert(trip_idx == sched.expanded_trips_.data_size());
  return trip_idx;
}

void add_footpaths(schedule const& sched, csa_timetable& tt) {
  for (auto& s : tt.stations_) {
    std::copy(begin(sched.stations_[s.id_]->outgoing_footpaths_),
              end(sched.stations_[s.id_]->outgoing_footpaths_),
              std::back_inserter(s.footpaths_));
    std::copy(begin(sched.stations_[s.id_]->incoming_footpaths_),
              end(sched.stations_[s.id_]->incoming_footpaths_),
              std::back_inserter(s.incoming_footpaths_));
  }
}

}  // namespace

std::unique_ptr<csa_timetable> build_csa_timetable(
    schedule const& sched, bool const bridge_zero_duration_connections,
    bool const add_footpath_connections) {
  scoped_timer timer("building csa timetable");

  auto tt = std::make_unique<csa_timetable>();

  // Create stations
  tt->stations_ = utl::to_vec(
      sched.stations_, [](auto const& st) { return csa_station(st.get()); });
  add_footpaths(sched, *tt);

  LOG(info) << "Creating CSA Connections";
  tt->trip_count_ = get_connections_from_expanded_trips(
      *tt, sched, bridge_zero_duration_connections, add_footpath_connections);

  init_trip_to_connections(*tt);
  init_stop_to_connections(*tt);

#ifdef MOTIS_CUDA
  {
    scoped_timer gpu_timer("building csa gpu timetable");
    tt->gpu_timetable_ = gpu_timetable(*tt);
  }
#endif

  LOG(info) << "CSA Stations: " << tt->stations_.size();
  LOG(info) << "CSA Connections: " << tt->fwd_connections_.size();

  return tt;
}

}  // namespace motis::csa
