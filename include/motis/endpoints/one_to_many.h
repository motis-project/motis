#pragma once

#include "boost/url/url_view.hpp"

#include "utl/to_vec.h"

#include "net/bad_request_exception.h"
#include "net/too_many_exception.h"

#include "osr/routing/route.h"

#include "motis-api/motis-api.h"
#include "motis/config.h"
#include "motis/fwd.h"
#include "motis/osr/mode_to_profile.h"
#include "motis/osr/parameters.h"
#include "motis/parse_location.h"

namespace motis::ep {

template <typename Params>
api::oneToMany_response one_to_many_handle_request(
    Params const& query,
    osr::ways const& w_,
    osr::lookup const& l_,
    osr::elevation_storage const* elevations_,
    unsigned const max_many,
    unsigned const max_travel_time_limit) {
  // required field with default value, not std::optional
  static_assert(std::is_same_v<decltype(query.withDistance_), bool>);

  auto const one = parse_location(query.one_, ';');
  utl::verify<net::bad_request_exception>(
      one.has_value(), "{} is not a valid geo coordinate", query.one_);

  auto const many = utl::to_vec(query.many_, [](auto&& x) {
    auto const y = parse_location(x, ';');
    utl::verify<net::bad_request_exception>(
        y.has_value(), "{} is not a valid geo coordinate", x);
    return *y;
  });
  utl::verify<net::too_many_exception>(
      many.size() <= max_many,
      "number of many locations too high ({} > {}). The server admin can "
      "change this limit in config.yml with 'onetomany_max_many'. "
      "See documentation for details.",
      many.size(), max_many);
  utl::verify<net::too_many_exception>(
      query.max_ <= max_travel_time_limit,
      "maximun travel time too high ({} > {}). The server admin can "
      "change this limit in config.yml with "
      "'street_routing_max_direct_seconds'. "
      "See documentation for details.",
      query.max_, max_travel_time_limit);

  utl::verify<net::bad_request_exception>(
      query.mode_ == api::ModeEnum::BIKE || query.mode_ == api::ModeEnum::CAR ||
          query.mode_ == api::ModeEnum::WALK,
      "mode {} not supported for one-to-many",
      boost::json::serialize(boost::json::value_from(query.mode_)));

  auto const profile = to_profile(query.mode_, api::PedestrianProfileEnum::FOOT,
                                  query.elevationCosts_);

  auto const paths = osr::route(
      to_profile_parameters(profile, get_osr_parameters(query)), w_, l_,
      profile, *one, many, query.max_,
      query.arriveBy_ ? osr::direction::kBackward : osr::direction::kForward,
      query.maxMatchingDistance_, nullptr, nullptr, elevations_,
      [&](auto&&) { return query.withDistance_; });

  return utl::to_vec(paths, [&](std::optional<osr::path> const& p) {
    return p
        .transform([&](osr::path const& x) {
          return api::Duration{.duration_ = x.cost_,
                               .distance_ = query.withDistance_
                                                ? std::optional{x.dist_}
                                                : std::nullopt};
        })
        .value_or(api::Duration{});
  });
}

struct one_to_many {
  api::oneToMany_response operator()(boost::urls::url_view const&) const;

  config const& config_;
  osr::ways const& w_;
  osr::lookup const& l_;
  osr::elevation_storage const* elevations_;
};

}  // namespace motis::ep
