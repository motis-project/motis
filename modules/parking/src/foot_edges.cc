#include <cmath>
#include <algorithm>
#include <functional>
#include <optional>
#include <string>

#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

#include "ppr/routing/search.h"
#include "ppr/serialization/reader.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/time.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_parallel_for.h"

#include "motis/parking/database.h"
#include "motis/parking/foot_edges.h"
#include "motis/parking/mumo_edges.h"
#include "motis/parking/thread_pool.h"

using namespace flatbuffers;
using namespace motis::logging;
using namespace ppr;
using namespace ppr::routing;
using namespace ppr::serialization;

namespace motis::parking {

struct foot_edge_info {
  duration duration_{};
  std::uint16_t accessibility_{};
  double distance_{};
};

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

std::vector<std::vector<foot_edge_info>> route_ppr_direct(
    routing_graph const& rg, location const& parking_loc,
    std::vector<location> const& station_locs, search_profile const& profile,
    search_direction const dir) {
  auto const result = find_routes(rg, parking_loc, station_locs, profile, dir);
  return utl::to_vec(result.routes_, [&](auto const& routes) {
    return utl::to_vec(routes, [&](auto const& r) {
      return foot_edge_info{get_duration(r), get_accessibility(r), r.distance_};
    });
  });
}

std::vector<std::vector<foot_edge_info>> route_ppr_module(
    location const& parking_loc, std::vector<location> const& station_locs,
    std::string const& profile_name, search_profile const& profile,
    search_direction const dir) {
  using namespace ppr;
  auto const req = make_ppr_request(
      parking_loc, station_locs, profile_name, profile.duration_limit_,
      dir == search_direction::FWD ? SearchDir_Forward : SearchDir_Backward);
  auto const msg = motis_call(req)->val();
  auto const ppr_res = motis_content(FootRoutingResponse, msg);
  return utl::to_vec(*ppr_res->routes(), [&](auto const& routes) {
    return utl::to_vec(*routes->routes(), [&](auto const& r) {
      return foot_edge_info{r->duration(), r->accessibility(), r->distance()};
    });
  });
}

using route_fn_t = std::function<std::vector<std::vector<foot_edge_info>>(
    location const& /*parking_loc*/,
    std::vector<location> const& /*station_locs*/,
    std::string const& /*profile_name*/, search_profile const& /*profile*/,
    search_direction /*dir*/)>;

void compute_edges(
    foot_edge_task const& task, database& db,
    std::map<std::string, motis::ppr::profile_info> const& ppr_profiles,
    bool const ppr_exact, route_fn_t const& route_fn) {
  auto const* lot = task.parking_lot_;
  auto const& profile_name = *task.ppr_profile_;
  auto const& profile = ppr_profiles.at(profile_name).profile_;
  FlatBufferBuilder fbb;
  std::vector<Offset<FootEdge>> outward_edges;
  std::vector<Offset<FootEdge>> return_edges;
  if (!task.stations_in_radius_.empty()) {
    auto const parking_loc = to_location(lot->location_);
    auto const station_locs = utl::to_vec(
        task.stations_in_radius_,
        [](auto const& s) { return to_location(std::get<0>(s).pos_); });

    auto const fwd_result = route_fn(parking_loc, station_locs, profile_name,
                                     profile, search_direction::FWD);
    assert(fwd_result.size() == task.stations_in_radius_.size());
    std::optional<std::vector<std::vector<foot_edge_info>>> bwd_result;
    if (ppr_exact) {
      bwd_result = route_fn(parking_loc, station_locs, profile_name, profile,
                            search_direction::BWD);
      assert(bwd_result->size() == task.stations_in_radius_.size());
    }

    for (auto station_idx = 0U; station_idx < task.stations_in_radius_.size();
         ++station_idx) {
      auto const& fwd_routes = fwd_result[station_idx];
      if (fwd_routes.empty() &&
          (!bwd_result.has_value() || (*bwd_result)[station_idx].empty())) {
        continue;
      }
      auto const station_id = fbb.CreateString(
          std::get<0>(task.stations_in_radius_[station_idx]).id_);
      auto const station_dist =
          std::get<1>(task.stations_in_radius_[station_idx]);
      for (auto const& r : fwd_routes) {
        outward_edges.emplace_back(CreateFootEdge(fbb, station_id, station_dist,
                                                  r.duration_, r.accessibility_,
                                                  r.distance_));
      }
      if (bwd_result.has_value()) {
        auto const bwd_routes = (*bwd_result)[station_idx];
        for (auto const& r : bwd_routes) {
          return_edges.emplace_back(
              CreateFootEdge(fbb, station_id, station_dist, r.duration_,
                             r.accessibility_, r.distance_));
        }
      }
    }
  }
  auto const fbs_outward_edges = fbb.CreateVector(outward_edges);
  auto const fbs_return_edges =
      ppr_exact ? fbb.CreateVector(return_edges) : fbs_outward_edges;
  fbb.Finish(CreateFootEdges(fbb, lot->id_,
                             fbb.CreateString(*task.ppr_profile_),
                             fbs_outward_edges, fbs_return_edges));
  db.put_footedges(persistable_foot_edges(std::move(fbb)),
                   task.stations_in_radius_);
}

void compute_foot_edges_direct(
    database& db, std::vector<foot_edge_task> const& tasks,
    motis::ppr::ppr_data const& ppr_data,
    std::map<std::string, motis::ppr::profile_info> const& ppr_profiles,
    int threads, bool ppr_exact) {
  LOG(info) << "Computing foot edges (" << tasks.size() << " tasks)...";

  scoped_timer const timer{"Computing foot edges"};

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->reset_bounds().in_high(tasks.size());
  thread_pool pool{static_cast<unsigned>(std::max(1, threads))};
  auto const route_fn = [&](location const& parking_loc,
                            std::vector<location> const& station_locs,
                            std::string const& /*profile_name*/,
                            search_profile const& profile,
                            search_direction const dir) {
    return route_ppr_direct(ppr_data.rg_, parking_loc, station_locs, profile,
                            dir);
  };
  for (auto const& t : tasks) {
    pool.post([&, &t = t] {
      progress_tracker->increment();
      compute_edges(t, db, ppr_profiles, ppr_exact, route_fn);
    });
  }
  pool.join();
  LOG(info) << "Foot edges precomputed.";
}

void compute_foot_edges_via_module(
    database& db, std::vector<foot_edge_task> const& tasks,
    std::map<std::string, motis::ppr::profile_info> const& ppr_profiles,
    bool ppr_exact) {
  scoped_timer const timer{"compute foot edges"};
  auto const route_fn = [&](location const& parking_loc,
                            std::vector<location> const& station_locs,
                            std::string const& profile_name,
                            search_profile const& profile,
                            search_direction const dir) {
    return route_ppr_module(parking_loc, station_locs, profile_name, profile,
                            dir);
  };
  motis_parallel_for(tasks, [&](foot_edge_task const& task) {
    compute_edges(task, db, ppr_profiles, ppr_exact, route_fn);
  });
}

}  // namespace motis::parking
