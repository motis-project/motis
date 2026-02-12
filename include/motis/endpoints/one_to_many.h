#pragma once

#include <memory>

#include "boost/url/url_view.hpp"

#include "utl/to_vec.h"

#include "nigiri/types.h"

#include "osr/location.h"
#include "osr/routing/route.h"
#include "osr/types.h"

#include "motis-api/motis-api.h"

#include "motis/data.h"
#include "motis/fwd.h"
#include "motis/match_platforms.h"
#include "motis/osr/parameters.h"
#include "motis/parse_location.h"
#include "motis/place.h"
#include "motis/point_rtree.h"

namespace motis::ep {

api::oneToMany_response one_to_many_direct(
    osr::ways const&,
    osr::lookup const&,
    api::ModeEnum const,
    osr::location const& one,
    std::vector<osr::location> const& many,
    double const max_direct_time,
    double const max_matching_distance,
    osr::direction,
    osr_parameters const&,
    api::PedestrianProfileEnum,
    api::ElevationCostsEnum,
    osr::elevation_storage const*);

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

  return one_to_many_direct(
      w_, l_, query.mode_, *one, many, query.max_, query.maxMatchingDistance_,
      query.arriveBy_ ? osr::direction::kBackward : osr::direction::kForward,
      get_osr_parameters(query), api::PedestrianProfileEnum::FOOT,
      query.elevationCosts_, elevations_);
}

template <typename Endpoint, typename Query>
api::oneToManyIntermodal_response run_one_to_many_intermodal(
    Endpoint const&,
    Query const&,
    place_t const&,
    std::vector<place_t> const& many);

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
