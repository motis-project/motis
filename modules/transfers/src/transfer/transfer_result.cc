#include "motis/transfers/transfer/transfer_result.h"

#include <cmath>
#include <algorithm>

#include "motis/transfers/platform/to_ppr.h"

#include "boost/thread/lock_types.hpp"
#include "boost/thread/mutex.hpp"

#include "fmt/core.h"

#include "ppr/routing/input_location.h"
#include "ppr/routing/route.h"
#include "ppr/routing/routing_query.h"
#include "ppr/routing/search.h"

#include "utl/parallel_for.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"
#include "utl/verify.h"
#include "utl/zip.h"

namespace n = ::nigiri;
namespace pr = ::ppr::routing;

namespace motis::transfers {

// Returns the duration of a given ppr::route.
inline n::duration_t get_duration(pr::route const& r) {
  return n::duration_t{static_cast<int>(std::round(r.duration_ / 60))};
}

// Generates and Returns a list auf `transfer_infos` build from the given
// ppr::search_result.
std::vector<transfer_infos> to_transfer_infos(pr::search_result const& res) {
  return utl::to_vec(res.routes_, [&](auto const& routes) {
    return utl::to_vec(routes, [&](auto const& r) {
      return transfer_info{get_duration(r), r.distance_};
    });
  });
}

// Builds a ppr::routing_query using the given `transfer_request` and a map of
// `profile_keys_t` to `profile_info` to get the search profile of the
// `transfer_request`.
pr::routing_query make_routing_query(
    hash_map<profile_key_t, ppr::profile_info> const& profiles,
    transfer_request const& treq) {
  // query: create start input_location
  auto const& li_start = to_input_location(treq.transfer_start_);

  // query: create dest input_locations
  std::vector<pr::input_location> ils_dests;
  std::transform(treq.transfer_targets_.cbegin(), treq.transfer_targets_.cend(),
                 std::back_inserter(ils_dests),
                 [](auto const& pf) { return to_input_location(pf); });

  // query: get search profile
  auto const& profile = profiles.at(treq.profile_).profile_;

  // query: get search direction (default: FWD)
  auto const& dir = pr::search_direction::FWD;

  return pr::routing_query(li_start, ils_dests, profile, dir);
}

transfer_result route_single_request(
    transfer_request const& treq, ::ppr::routing_graph const& rg,
    hash_map<profile_key_t, ppr::profile_info> const& profiles) {
  auto tres = transfer_result{};
  tres.from_nloc_key_ = treq.from_nloc_key_;
  tres.profile_ = treq.profile_;

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

    if (fwd_routes.empty()) {
      continue;
    }

    tres.to_nloc_keys_.emplace_back(treq.to_nloc_keys_[i]);
    tres.infos_.emplace_back(fwd_routes.front());
  }

  return tres;
}

transfer_results route_multiple_requests(
    transfer_requests const& treqs, ::ppr::routing_graph const& rg,
    hash_map<profile_key_t, ppr::profile_info> const& profiles) {
  auto result = transfer_results{};

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->in_high(treqs.size());

  boost::mutex mutex;
  utl::parallel_for(treqs, [&](auto const& treq) {
    auto single_result = route_single_request(treq, rg, profiles);
    {
      boost::unique_lock<boost::mutex> const scoped_lock(mutex);
      result.emplace_back(single_result);
    }
    progress_tracker->increment();
  });

  return result;
}

transfer_result merge(transfer_result const& tres_a,
                      transfer_result const& tres_b) {
  auto merged = transfer_result{};
  auto added_to_nlocs = set<nlocation_key_t>{};

  utl::verify(tres_a.from_nloc_key_ == tres_b.from_nloc_key_,
              "Cannot merge two transfer results from different locations.");
  utl::verify(tres_a.profile_ == tres_b.profile_,
              "Cannot merge two transfer results with different profiles.");
  utl::verify(tres_a.to_nloc_keys_.size() == tres_a.infos_.size(),
              "(A) Cannot merge transfer results with invalid target and info "
              "matching.");
  utl::verify(tres_b.to_nloc_keys_.size() == tres_b.infos_.size(),
              "(B) Cannot merge transfer results with invalid target and info "
              "matching.");

  merged.from_nloc_key_ = tres_a.from_nloc_key_;
  merged.profile_ = tres_a.profile_;

  merged.to_nloc_keys_ = tres_a.to_nloc_keys_;
  merged.infos_ = tres_a.infos_;

  // build added_to_nlocs set
  for (auto const& nloc_key : merged.to_nloc_keys_) {
    added_to_nlocs.insert(nloc_key);
  }

  // insert new and unique nloc/info keys
  for (auto const [nloc_key, info] :
       utl::zip(tres_b.to_nloc_keys_, tres_b.infos_)) {
    if (added_to_nlocs.count(nloc_key) == 1) {
      continue;
    }

    merged.to_nloc_keys_.emplace_back(nloc_key);
    merged.infos_.emplace_back(info);
    added_to_nlocs.insert(nloc_key);
  }

  return merged;
}

string to_key(transfer_result const& tres) {
  return {fmt::format("{}{}", tres.from_nloc_key_, tres.profile_)};
}

std::ostream& operator<<(std::ostream& out, transfer_info const& tinfo) {
  return out << "dur: " << tinfo.duration_ << ", dist: " << tinfo.distance_;
}

std::ostream& operator<<(std::ostream& out, transfer_result const& tres) {
  std::stringstream tres_repr;
  tres_repr << "[transfer result] " << to_key(tres) << ": #results - "
            << tres.infos_.size();
  return out << tres_repr.str();
}

}  // namespace motis::transfers
