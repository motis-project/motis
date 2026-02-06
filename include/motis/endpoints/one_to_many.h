#pragma once

#include <memory>

#include "boost/url/url_view.hpp"

#include "utl/to_vec.h"

#include "osr/routing/route.h"

#include "nigiri/types.h"

#include "motis-api/motis-api.h"

#include "motis/data.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/osr/mode_to_profile.h"
#include "motis/osr/parameters.h"
#include "motis/parse_location.h"
#include "motis/point_rtree.h"

namespace motis::ep {

template <typename Params>
api::oneToMany_response one_to_many_handle_request(
    Params const& query,
    osr::ways const& w_,
    osr::lookup const& l_,
    osr::elevation_storage const* elevations_) {
  auto const one = parse_location(query.one_, ';');
  utl::verify(one.has_value(), "{} is not a valid geo coordinate", query.one_);

  auto const many = utl::to_vec(query.many_, [](auto&& x) {
    auto const y = parse_location(x, ';');
    utl::verify(y.has_value(), "{} is not a valid geo coordinate", x);
    return *y;
  });

  utl::verify(query.mode_ == api::ModeEnum::BIKE ||
                  query.mode_ == api::ModeEnum::CAR ||
                  query.mode_ == api::ModeEnum::WALK,
              "mode {} not supported for one-to-many",
              boost::json::serialize(boost::json::value_from(query.mode_)));

  auto const profile = to_profile(query.mode_, api::PedestrianProfileEnum::FOOT,
                                  query.elevationCosts_);
  auto const paths = osr::route(
      to_profile_parameters(profile, get_osr_parameters(query)), w_, l_,
      profile, *one, many, query.max_,
      query.arriveBy_ ? osr::direction::kBackward : osr::direction::kForward,
      query.maxMatchingDistance_, nullptr, nullptr, elevations_);

  return utl::to_vec(paths, [](std::optional<osr::path> const& p) {
    return p.has_value() ? api::Duration{.duration_ = p->cost_}
                         : api::Duration{};
  });
}

template <typename Endpoint, typename Query>
api::oneToManyIntermodal_response run_one_to_many_intermodal(
    Endpoint const& ep, Query const& query);

struct one_to_many {
  api::oneToMany_response operator()(boost::urls::url_view const&) const;

  osr::ways const& w_;
  osr::lookup const& l_;
  osr::elevation_storage const* elevations_;
};

struct one_to_many_intermodal {
  api::oneToManyIntermodal_response operator()(
      boost::urls::url_view const&) const;

  config const& config_;
  osr::ways const* w_;
  osr::lookup const* l_;
  osr::platforms const* pl_;
  osr::elevation_storage const* elevations_;
  nigiri::timetable const& tt_;
  std::shared_ptr<rt> const& rt_;
  tag_lookup const& tags_;
  flex::flex_areas const* fa_;
  point_rtree<nigiri::location_idx_t> const* loc_tree_;
  platform_matches_t const* matches_;
  way_matches_storage const* way_matches_;
  std::shared_ptr<gbfs::gbfs_data> const& gbfs_;
  metrics_registry* metrics_;
};

}  // namespace motis::ep
