#include "motis/footpaths/transfer_updates.h"

#include <cmath>

#include "cista/containers/vector.h"
#include "cista/containers/vecvec.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/time.h"

#include "motis/footpaths/thread_pool.h"

#include "ppr/routing/search.h"

#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

using namespace motis::logging;
using namespace ppr;
using namespace ppr::routing;

namespace n = nigiri;

namespace motis::footpaths {

struct transfer_edge_info {
  n::duration_t duration_{};
  std::uint16_t accessibility_{};
  double distance_{};
};

inline location to_location(geo::latlng const& pos) {
  return make_location(pos.lng_, pos.lat_);
}

inline n::duration_t get_duration(route const& r) {
  return n::duration_t{
      std::min(static_cast<int>(std::round(r.duration_ / 60)),
               static_cast<int>(std::numeric_limits<duration>::max()))};
}

inline uint16_t get_accessibility(route const& r) {
  return static_cast<uint16_t>(std::ceil(r.accessibility_));
}

/**
 * Returns the equivalent OSM_TYPE of PPR.
 * Default: NODE
 */
osm_namespace to_ppr_osm_type(nigiri::osm_type const& t) {
  switch (t) {
    case nigiri::osm_type::NODE: return osm_namespace::NODE;
    case nigiri::osm_type::WAY: return osm_namespace::WAY;
    case nigiri::osm_type::RELATION: return osm_namespace::RELATION;
    default: return osm_namespace::NODE;
  }
}

/**
 * Creates an input-location struct from a platform-info struct.
 *
 * @param pi the platform-info from which to create an input location.
 * @return an input location struct
 */
input_location pi_to_il(platform_info const& pi) {
  input_location il;
  // TODO (Carsten) OSM_ELEMENT LEVEL missing
  il.osm_element_ = {pi.osm_id_, to_ppr_osm_type(pi.osm_type_)};
  il.location_ = to_location(pi.pos_);
  return il;
}

/**
 * Creates a routing_query from a transfer_request.
 *
 * @param req the transfer-request from which to create a routing query.
 * @return a routing query struct.
 */
routing_query make_routing_query(
    std::map<std::string, ppr::profile_info> const& ppr_profiles,
    transfer_requests const& t_req) {
  // query: create start input_location
  auto const& li_start = pi_to_il(*t_req.transfer_start_);

  // query: create dest input_locations
  std::vector<input_location> ils_dests;
  std::transform(t_req.transfer_targets_.cbegin(),
                 t_req.transfer_targets_.cend(), std::back_inserter(ils_dests),
                 [](auto const& pi) { return pi_to_il(*pi); });

  // query: get search profile
  auto const& profile = ppr_profiles.at(t_req.profile_name).profile_;

  // query: get search direction (default: FWD)
  auto const& dir = search_direction::FWD;

  return routing_query(li_start, ils_dests, profile, dir);
}

search_result route_ppr_direct(
    routing_graph const& rg,
    std::map<std::string, ppr::profile_info> const& ppr_profiles,
    transfer_requests const& t_req) {
  auto const& rq = make_routing_query(ppr_profiles, t_req);

  // route using find_routes_v2
  return find_routes_v2(rg, rq);
}

std::vector<std::vector<transfer_edge_info>> to_transfer_edge_info(
    search_result const& res) {
  return utl::to_vec(res.routes_, [&](auto const& routes) {
    return utl::to_vec(routes, [&](auto const& r) {
      return transfer_edge_info{get_duration(r), get_accessibility(r),
                                r.distance_};
    });
  });
}

void compute_and_update_nigiri_transfers(
    routing_graph const& rg, nigiri::timetable& tt,
    std::map<std::string, ppr::profile_info> const& ppr_profiles,
    transfer_requests const& t_req, boost::mutex& mutex) {
  auto const& res = route_ppr_direct(rg, ppr_profiles, t_req);

  if (res.destinations_reached() == 0) {
    return;
  }

  auto const& fwd_result = to_transfer_edge_info(res);
  assert(fwd_result.size() == t_req.transfer_targets_.size());

  for (auto platform_idx = 0U; platform_idx < t_req.transfer_targets_.size();
       ++platform_idx) {
    auto const& fwd_routes = fwd_result[platform_idx];
    if (fwd_routes.empty()) {
      continue;
    }

    for (auto const& r : fwd_routes) {
      boost::unique_lock<boost::mutex> const scoped_lock(mutex);

      auto const& profile_idx = tt.locations_.profile_idx_[t_req.profile_name];

      tt.locations_.footpaths_out_[profile_idx][t_req.transfer_start_->idx_]
          .push_back(n::footpath{t_req.transfer_targets_[platform_idx]->idx_,
                                 r.duration_});
      tt.locations_
          .footpaths_out_[profile_idx]
                         [t_req.transfer_targets_[platform_idx]->idx_]
          .push_back(n::footpath{t_req.transfer_start_->idx_, r.duration_});
    }
  }
}

void precompute_nigiri_transfers(
    routing_graph const& rg, nigiri::timetable& tt,
    std::map<std::string, ppr::profile_info> const& ppr_profiles,
    std::vector<transfer_requests> const& transfer_reqs) {
  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->reset_bounds().in_high(transfer_reqs.size());

  thread_pool pool{std::max(1U, std::thread::hardware_concurrency())};
  boost::mutex mutex;
  for (auto const& t_req : transfer_reqs) {
    pool.post([&, &t_req = t_req] {
      progress_tracker->increment();
      compute_and_update_nigiri_transfers(rg, tt, ppr_profiles, t_req, mutex);
    });
  }
  pool.join();
  LOG(info) << "Profilebased transfers precomputed.";
};

}  // namespace motis::footpaths