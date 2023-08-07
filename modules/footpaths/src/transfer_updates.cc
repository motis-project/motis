#include "motis/footpaths/transfer_updates.h"

#include <cmath>

#include "motis/core/common/logging.h"
#include "motis/core/schedule/time.h"

#include "motis/footpaths/platforms.h"

#include "utl/parallel_for.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

namespace ml = motis::logging;
namespace n = nigiri;
namespace p = ppr;
namespace pr = ppr::routing;

namespace motis::footpaths {

struct transfer_edge_info {
  n::duration_t duration_{};
  std::uint16_t accessibility_{};
  double distance_{};
};

inline n::duration_t get_duration(pr::route const& r) {
  return n::duration_t{
      std::min(static_cast<int>(std::round(r.duration_ / 60)),
               static_cast<int>(std::numeric_limits<duration>::max()))};
}

inline uint16_t get_accessibility(pr::route const& r) {
  return static_cast<uint16_t>(std::ceil(r.accessibility_));
}

/**
 * Creates a routing_query from a transfer_request.
 *
 * @param req the transfer-request from which to create a routing query.
 * @return a routing query struct.
 */
pr::routing_query make_routing_query(
    std::map<std::string, ppr::profile_info> const& ppr_profiles,
    transfer_requests const& t_req) {
  // query: create start input_location
  auto const& li_start = to_input_location(*t_req.transfer_start_);

  // query: create dest input_locations
  std::vector<pr::input_location> ils_dests;
  std::transform(t_req.transfer_targets_.cbegin(),
                 t_req.transfer_targets_.cend(), std::back_inserter(ils_dests),
                 [](auto const& pf) { return to_input_location(*pf); });

  // query: get search profile
  auto const& profile = ppr_profiles.at(t_req.profile_name).profile_;

  // query: get search direction (default: FWD)
  auto const& dir = pr::search_direction::FWD;

  return pr::routing_query(li_start, ils_dests, profile, dir);
}

pr::search_result route_ppr_direct(
    p::routing_graph const& rg,
    std::map<std::string, ppr::profile_info> const& ppr_profiles,
    transfer_requests const& t_req) {
  auto const& rq = make_routing_query(ppr_profiles, t_req);

  // route using find_routes_v2
  return find_routes_v2(rg, rq);
}

std::vector<std::vector<transfer_edge_info>> to_transfer_edge_info(
    pr::search_result const& res) {
  return utl::to_vec(res.routes_, [&](auto const& routes) {
    return utl::to_vec(routes, [&](auto const& r) {
      return transfer_edge_info{get_duration(r), get_accessibility(r),
                                r.distance_};
    });
  });
}

void compute_and_update_nigiri_transfers(
    p::routing_graph const& rg, nigiri::timetable& tt,
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

      auto const& profile_idx = tt.profiles_[t_req.profile_name];

      tt.locations_
          .footpaths_out_[profile_idx][t_req.transfer_start_->info_.idx_]
          .push_back(n::footpath{
              t_req.transfer_targets_[platform_idx]->info_.idx_, r.duration_});
      tt.locations_
          .footpaths_in_[profile_idx]
                        [t_req.transfer_targets_[platform_idx]->info_.idx_]
          .push_back(
              n::footpath{t_req.transfer_start_->info_.idx_, r.duration_});
    }
  }
}

void precompute_nigiri_transfers(
    p::routing_graph const& rg, nigiri::timetable& tt,
    std::map<std::string, ppr::profile_info> const& ppr_profiles,
    std::vector<transfer_requests> const& transfer_reqs) {
  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->increment(transfer_reqs.size());

  boost::mutex mutex;
  utl::parallel_for(transfer_reqs, [&](auto const& t_req) {
    compute_and_update_nigiri_transfers(rg, tt, ppr_profiles, t_req, mutex);
    progress_tracker->increment();
  });

  LOG(ml::info) << "Profilebased transfers precomputed.";
};

}  // namespace motis::footpaths