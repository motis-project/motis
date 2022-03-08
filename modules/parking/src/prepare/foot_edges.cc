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
    foot_edge_task const& task, database& db, routing_graph const& rg,
    std::map<std::string, motis::ppr::profile_info> const& ppr_profiles) {
  auto const* lot = task.parking_lot_;
  auto const& profile = ppr_profiles.at(*task.ppr_profile_).profile_;
  FlatBufferBuilder fbb;
  std::vector<Offset<FootEdge>> outward_edges;
  std::vector<Offset<FootEdge>> return_edges;
  if (!task.stations_in_radius_.empty()) {
    auto const parking_loc = to_location(lot->location_);
    auto const station_locs = utl::to_vec(
        task.stations_in_radius_,
        [](auto const& s) { return to_location(std::get<0>(s).pos_); });
    auto const fwd_result = find_routes(rg, parking_loc, station_locs, profile,
                                        search_direction::FWD);
    auto const bwd_result = find_routes(rg, parking_loc, station_locs, profile,
                                        search_direction::BWD);

    assert(fwd_result.routes_.size() == parking_stations.size());
    assert(bwd_result.routes_.size() == parking_stations.size());
    for (auto station_idx = 0U; station_idx < task.stations_in_radius_.size();
         ++station_idx) {
      auto const fwd_routes = fwd_result.routes_[station_idx];
      auto const bwd_routes = bwd_result.routes_[station_idx];
      if (fwd_routes.empty() && bwd_routes.empty()) {
        continue;
      }
      auto const station_id = fbb.CreateString(
          std::get<0>(task.stations_in_radius_[station_idx]).id_);
      auto const station_dist =
          std::get<1>(task.stations_in_radius_[station_idx]);
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
  fbb.Finish(CreateFootEdges(
      fbb, lot->id_, fbb.CreateString(*task.ppr_profile_),
      fbb.CreateVector(outward_edges), fbb.CreateVector(return_edges)));
  db.put_footedges(persistable_foot_edges(std::move(fbb)),
                   task.stations_in_radius_);
}

void compute_foot_edges(
    database& db, std::vector<foot_edge_task> const& tasks,
    std::map<std::string, motis::ppr::profile_info> const& ppr_profiles,
    std::string const& ppr_graph, std::size_t edge_rtree_max_size,
    std::size_t area_rtree_max_size, bool lock_rtrees, int threads) {
  LOG(info) << "Computing foot edges (" << tasks.size() << " tasks)...";

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

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->reset_bounds().in_high(tasks.size());
  thread_pool pool{static_cast<unsigned>(std::max(1, threads))};
  for (auto const& [i, t] : utl::enumerate(tasks)) {
    pool.post([&, i = i, &t = t] {
      progress_tracker->increment();
      compute_edges(t, db, rg, ppr_profiles);
    });
  }
  pool.join();
  LOG(info) << "Foot edges precomputed.";
}

}  // namespace motis::parking::prepare
