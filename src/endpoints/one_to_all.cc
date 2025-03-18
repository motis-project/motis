#include "motis/endpoints/one_to_all.h"

#include <chrono>
#include <vector>

#include "nigiri/common/delta_t.h"
#include "nigiri/location_match_mode.h"
#include "nigiri/routing/one_to_all.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/place.h"
#include "motis/tag_lookup.h"

namespace motis::ep {
api::Reachable one_to_all::operator()(boost::urls::url_view const& url) const {
  auto const query = api::oneToAll_params{url.params()};

  auto const unreachable =
      query.arriveBy_ ? nigiri::kInvalidDelta<nigiri::direction::kBackward>
                      : nigiri::kInvalidDelta<nigiri::direction::kForward>;

  auto const time = std::chrono::time_point_cast<std::chrono::minutes>(
      *query.time_.value_or(openapi::now()));
  auto const l = tags_.get_location(tt_, query.one_);
  auto const pos = tt_.locations_.coordinates_[l];

  auto const q = nigiri::routing::query{
      .start_time_ = time,
      .start_match_mode_ = nigiri::routing::location_match_mode::kEquivalent,
      .start_ = {{l, nigiri::duration_t{}, nigiri::transport_mode_id_t{0U}}},
      .max_travel_time_ = nigiri::duration_t{query.max_},
  };

  auto const state = [&]() {
    if (query.arriveBy_) {
      return nigiri::routing::one_to_all<nigiri::direction::kBackward>(
          tt_, nullptr, q);
    } else {
      return nigiri::routing::one_to_all<nigiri::direction::kForward>(
          tt_, nullptr, q);
    }
  }();

  auto const one = api::Place{
      .name_ = std::string{tt_.locations_.names_[l].view()},
      .stopId_ = tags_.id(tt_, l),
      .lat_ = pos.lat(),
      .lon_ = pos.lng(),
      .level_ = static_cast<double>(to_idx(get_lvl(w_, pl_, matches_, l))),
      .departure_ = time,
  };

  auto all = std::vector<api::ReachablePlace>{};
  for (auto i = nigiri::location_idx_t{0U}; i < tt_.n_locations(); ++i) {
    if (state.get_best<0>()[to_idx(i)][0] != unreachable) {
      auto const dst = tt_.locations_.coordinates_[i];
      auto const fastest = [&]() {
        if (query.arriveBy_) {
          return nigiri::routing::get_fastest_one_to_all_offsets<
              nigiri::direction::kBackward>(tt_, state, i, time,
                                            q.max_transfers_);
        } else {
          return nigiri::routing::get_fastest_one_to_all_offsets<
              nigiri::direction::kForward>(tt_, state, i, time,
                                           q.max_transfers_);
        }
      }();

      all.emplace_back(
          api::Place{
              .name_ = std::string{tt_.locations_.names_[i].view()},
              .stopId_ = tags_.id(tt_, i),
              .lat_ = dst.lat(),
              .lon_ = dst.lng(),
              .level_ =
                  static_cast<double>(to_idx(get_lvl(w_, pl_, matches_, i))),
              .arrival_ = time + std::chrono::minutes{fastest.duration_},
          },
          query.arriveBy_ ? -fastest.duration_ : fastest.duration_, fastest.k_);
    }
  }
  return {
      .one_ = std::move(one),
      .all_ = std::move(all),
  };
}

}  // namespace motis::ep
