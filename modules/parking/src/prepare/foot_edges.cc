#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>

#include "utl/enumerate.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

#include "ppr/routing/search.h"
#include "ppr/serialization/reader.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/time.h"

#include "motis/parking/database.h"
#include "motis/parking/prepare/foot_edges.h"
#include "motis/parking/prepare/thread_pool.h"

using namespace flatbuffers;
using namespace motis::logging;
using namespace ppr;
using namespace ppr::routing;
using namespace ppr::serialization;

namespace motis::parking::prepare {

inline location to_location(geo::latlng const& pos) {
  return make_location(pos.lng_, pos.lat_);
}

inline duration get_duration(route const& r) {
  return static_cast<duration>(
      std::min(std::round(r.duration_ / 60),
               static_cast<double>(std::numeric_limits<duration>::max())));
}

inline uint16_t get_accessibility(route const& r) {
  return static_cast<uint16_t>(std::ceil(r.accessibility_));
}

void compute_edges(
    database& db, routing_graph const& rg, stations const& st,
    std::map<std::string, motis::ppr::profile_info> const& ppr_profiles,
    parking_lot const& p, std::vector<unsigned>& stations_per_parking,
    size_t work_idx) {
  for (auto const& [profile_name, pi] : ppr_profiles) {
    auto const& profile = pi.profile_;
    FlatBufferBuilder fbb;
    std::vector<Offset<FootEdge>> outward_edges;
    std::vector<Offset<FootEdge>> return_edges;
    auto const walk_radius = static_cast<int>(
        std::ceil(profile.duration_limit_ * profile.walking_speed_));
    auto const parking_stations = st.get_in_radius(p.location_, walk_radius);
    stations_per_parking[work_idx] =
        std::max(stations_per_parking[work_idx],
                 static_cast<unsigned>(parking_stations.size()));
    if (!parking_stations.empty()) {
      auto const parking_loc = to_location(p.location_);
      auto const station_locs = utl::to_vec(
          parking_stations,
          [](auto const& s) { return to_location(std::get<0>(s).pos_); });
      auto const fwd_result = find_routes(rg, parking_loc, station_locs,
                                          profile, search_direction::FWD);
      auto const bwd_result = find_routes(rg, parking_loc, station_locs,
                                          profile, search_direction::BWD);

      assert(fwd_result.routes_.size() == parking_stations.size());
      assert(bwd_result.routes_.size() == parking_stations.size());
      for (auto station_idx = 0U; station_idx < parking_stations.size();
           ++station_idx) {
        auto const fwd_routes = fwd_result.routes_[station_idx];
        auto const bwd_routes = bwd_result.routes_[station_idx];
        if (fwd_routes.empty() && bwd_routes.empty()) {
          continue;
        }
        auto const station_id =
            fbb.CreateString(std::get<0>(parking_stations[station_idx]).id_);
        auto const station_dist = std::get<1>(parking_stations[station_idx]);
        for (auto const& r : fwd_routes) {
          outward_edges.emplace_back(
              CreateFootEdge(fbb, station_id, station_dist, get_duration(r),
                             get_accessibility(r), r.distance_));
        }
        for (auto const& r : bwd_routes) {
          return_edges.emplace_back(
              CreateFootEdge(fbb, station_id, station_dist, get_duration(r),
                             get_accessibility(r), r.distance_));
        }
      }
    }
    fbb.Finish(CreateFootEdges(fbb, p.id_, fbb.CreateString(profile_name),
                               fbb.CreateVector(outward_edges),
                               fbb.CreateVector(return_edges)));
    db.put(persistable_foot_edges(std::move(fbb)));
  }
}

void compute_foot_edges(
    stations const& st, parkings const& park,
    std::string const& footedges_db_file, std::string const& ppr_graph,
    std::size_t edge_rtree_max_size, std::size_t area_rtree_max_size,
    bool lock_rtrees,
    std::map<std::string, motis::ppr::profile_info> const& ppr_profiles,
    int threads, std::string const& stations_per_parking_file) {
  std::clog << "Computing foot edges..." << std::endl;
  database db{footedges_db_file,
              static_cast<std::size_t>(1024) * 1024 * 1024 * 512, false};

  routing_graph rg;
  {
    scoped_timer ppr_load_timer{"Loading ppr routing graph"};
    read_routing_graph(rg, ppr_graph);
  }
  {
    scoped_timer ppr_rtree_timer{"Preparing ppr r-trees"};
    rg.prepare_for_routing(
        edge_rtree_max_size, area_rtree_max_size,
        lock_rtrees ? rtree_options::LOCK : rtree_options::PREFETCH);
  }

  scoped_timer timer{"Computing foot edges"};

  thread_pool pool{static_cast<unsigned>(std::max(1, threads))};

  auto max = park.parkings_.size();
  std::vector<unsigned> stations_per_parking;
  stations_per_parking.resize(max);
  std::clog << "Precomputing foot edges for " << max << " parkings..."
            << std::endl;

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->reset_bounds().in_high(max);
  for (auto const& [i, p] : utl::enumerate(park.parkings_)) {
    pool.post([&, i = i, &p = p] {
      progress_tracker->increment();
      compute_edges(db, rg, st, ppr_profiles, p, stations_per_parking, i);
    });
  }
  pool.join();
  std::clog << "Foot edges precomputed." << std::endl;

  {
    std::ofstream f(stations_per_parking_file);
    for (auto const n : stations_per_parking) {
      f << n << "\n";
    }
  }
}

}  // namespace motis::parking::prepare
