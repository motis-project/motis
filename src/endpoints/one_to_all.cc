#include "motis/endpoints/one_to_all.h"

#include <chrono>
#include <iostream>  // TODO Remove

#include "motis-api/motis-api.h"
#include "utl/verify.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/one_to_all.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "motis/endpoints/routing.h"
#include "motis/gbfs/routing_data.h"
#include "motis/parse_location.h"

namespace json = boost::json;

namespace motis::ep {
api::Reachable one_to_all::operator()(
    boost::urls::url_view const& url) const {
        auto const query = api::oneToAll_params{url.params()};
    std::cout << "URL: " << url << "\n";
    std::cout << query.one_ << ", " << query.max_ << "\n";

  auto const one = parse_location(query.one_, ';');
  utl::verify(one.has_value(), "{} is not a valid geo coordinate", query.one_);

  auto gbfs_rd = gbfs::gbfs_routing_data{w_, l_, gbfs_};
  // auto gbfs_rd = gbfs::gbfs_routing_data{&w_, &l_, gbfs_};

  // auto const dir = query.arriveBy_ ? osr::direction::kBackward
  //                                                  : osr::direction::kForward;
  auto const time = std::chrono::time_point_cast<std::chrono::minutes>(*query.time_.value_or(openapi::now()));
  auto const start = [&](osr::location const& pos) {

                  auto const dir = query.arriveBy_ ? osr::direction::kBackward
                                                   : osr::direction::kForward;
                  // return routing::get_offsets(
                  auto const r = routing{config_, w_, l_, pl_, tt_, tags_, loc_tree_, matches_, rt_, shapes_, gbfs_, odm_bounds_};
                  return r.get_offsets(
                      pos, dir, /*start_modes*/ {}, /*start_form_factors*/ {},
                      /*start_propulsion_types*/ {}, /*start_rental_providers*/ {},
                      query.pedestrianProfile_ ==
                          api::PedestrianProfileEnum::WHEELCHAIR,
                      std::chrono::seconds{query.maxPreTransitTime_},
                      query.maxMatchingDistance_, gbfs_rd);
  }(*one);
  for (auto const s : start) {
    std::cout << "Start: " << s.target() << ", " << s.duration() << "\n";
  }
  auto const q = nigiri::routing::query{
    .start_time_ = time,
      .start_ = std::move(start),
      .max_travel_time_ = nigiri::duration_t{query.max_},
  };
  // auto const state = nigiri::routing::one_to_all<dir>(tt_, rtt_, q);
  auto const state = [&]() {
    if (query.arriveBy_) {
      return nigiri::routing::one_to_all<nigiri::direction::kBackward>(*tt_, nullptr, q);
    } else {
      return nigiri::routing::one_to_all<nigiri::direction::kForward>(*tt_, nullptr, q);
    }
  }();

  // auto const x = one.
  auto const p = api::Place{
    .name_ = query.one_,
    .lat_ = one->pos_.lat(),
    .lon_ = one->pos_.lng(),
    .level_ = static_cast<double>(to_idx(one->lvl_)),
    .departure_ = time,
  };
  auto count = 0;
  for (auto i = 0U; i < tt_->n_locations(); ++i) {
    if (state.get_best<0>()[i][0] != nigiri::kInvalidDelta<nigiri::direction::kForward>) ++count;
  }
  std::cout << "Counted: " << count << "\n";
    return {
      .one_ = std::move(p),
    };
    }

}
