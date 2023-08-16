#include "motis/footpaths/transfer_results.h"

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <iterator>

#include "boost/thread/lock_types.hpp"
#include "boost/thread/mutex.hpp"

#include "motis/footpaths/platforms.h"

#include "nigiri/types.h"

#include "ppr/routing/input_location.h"
#include "ppr/routing/route.h"
#include "ppr/routing/routing_query.h"
#include "ppr/routing/search.h"

#include "utl/parallel_for.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"

namespace n = nigiri;
namespace pr = ppr::routing;

namespace motis::footpaths {

inline n::duration_t get_duration(pr::route const& r) {
  return n::duration_t{static_cast<int>(std::round(r.duration_ / 60))};
}

std::vector<transfer_infos> to_transfer_infos(pr::search_result const& res) {
  return utl::to_vec(res.routes_, [&](auto const& routes) {
    return utl::to_vec(routes, [&](auto const& r) {
      return transfer_info{get_duration(r), r.distance_};
    });
  });
}

pr::routing_query make_routing_query(
    std::map<std::string, ppr::profile_info> const& ppr_profiles,
    transfer_request const& tr) {
  // query: create start input_location
  auto const& li_start = to_input_location(tr.transfer_start_);

  // query: create dest input_locations
  std::vector<pr::input_location> ils_dests;
  std::transform(tr.transfer_targets_.cbegin(), tr.transfer_targets_.cend(),
                 std::back_inserter(ils_dests),
                 [](auto const& pf) { return to_input_location(pf); });

  // query: get search profile
  auto const& profile = ppr_profiles.at(tr.profile_).profile_;

  // query: get search direction (default: FWD)
  auto const& dir = pr::search_direction::FWD;

  return pr::routing_query(li_start, ils_dests, profile, dir);
}

transfer_results route_single_request(
    transfer_request const& treq, ::ppr::routing_graph const& rg,
    std::map<std::string, ppr::profile_info> const& profiles) {
  auto tress = transfer_results{};
  auto const& rq = make_routing_query(profiles, treq);

  // route using find_routes_v2
  auto const& search_res = pr::find_routes_v2(rg, rq);

  if (search_res.destinations_reached() == 0) {
    return {};
  }

  auto const& fwd_result = to_transfer_infos(search_res);
  assert(fwd_result.size() == treq.transfer_targets_.size());

  for (auto i = std::size_t{0}; i < treq.transfer_targets_.size(); ++i) {
    auto const& fwd_routes = fwd_result[i];
    auto result = transfer_result{};

    if (fwd_routes.empty()) {
      continue;
    }

    result.from_nloc_key_ = treq.from_nloc_key_;
    result.to_nloc_key_ = treq.to_nloc_keys_[i];
    result.profile_ = treq.profile_;
    result.info_ = fwd_routes.front();

    tress.emplace_back(result);
  }

  return tress;
}

transfer_results route_multiple_requests(
    transfer_requests const& treqs, ::ppr::routing_graph const& rg,
    std::map<std::string, ppr::profile_info> const& profiles) {
  auto result = transfer_results{};

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->in_high(treqs.size());

  boost::mutex mutex;
  utl::parallel_for(treqs, [&](auto const& treq) {
    auto single_result = route_single_request(treq, rg, profiles);
    {
      boost::unique_lock<boost::mutex> const scoped_lock(mutex);
      result.insert(result.end(), single_result.begin(), single_result.end());
    }
    progress_tracker->increment();
  });

  return result;
}

}  // namespace motis::footpaths
