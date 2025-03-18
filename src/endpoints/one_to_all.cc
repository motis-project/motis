#include "motis/endpoints/one_to_all.h"

#include <chrono>
#include <vector>

#include "utl/verify.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/location_match_mode.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/one_to_all.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/place.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/modes_to_clasz_mask.h"

namespace motis::ep {

api::Reachable one_to_all::operator()(boost::urls::url_view const& url) const {
  auto const query = api::oneToAll_params{url.params()};
  if (query.maxTransfers_.has_value()) {
    utl::verify(query.maxTransfers_ >= 0U, "maxTransfers < 0: {}",
                *query.maxTransfers_);
    utl::verify(query.maxTransfers_ <= nigiri::routing::kMaxTransfers,
                "maxTransfers > {}: {}", nigiri::routing::kMaxTransfers,
                *query.maxTransfers_);
  }

  auto const unreachable =
      query.arriveBy_ ? nigiri::kInvalidDelta<nigiri::direction::kBackward>
                      : nigiri::kInvalidDelta<nigiri::direction::kForward>;

  auto const time = std::chrono::time_point_cast<std::chrono::minutes>(
      *query.time_.value_or(openapi::now()));
  auto const l = tags_.get_location(tt_, query.one_);

  auto const q = nigiri::routing::query{
      .start_time_ = time,
      .start_match_mode_ = nigiri::routing::location_match_mode::kEquivalent,
      .start_ = {{l, nigiri::duration_t{}, nigiri::transport_mode_id_t{0U}}},
      .max_transfers_ = static_cast<std::uint8_t>(
          query.maxTransfers_.value_or(nigiri::routing::kMaxTransfers)),
      .max_travel_time_ = nigiri::duration_t{query.maxTravelTime_},
      .prf_idx_ = static_cast<nigiri::profile_idx_t>(
          query.useRoutedTransfers_
              ? (query.pedestrianProfile_ ==
                         api::PedestrianProfileEnum::WHEELCHAIR
                     ? 2U
                     : 1U)
              : 0U),
      .allowed_claszes_ = to_clasz_mask(query.transitModes_),
      .require_bike_transport_ = query.requireBikeTransport_,
      .transfer_time_settings_ =
          nigiri::routing::transfer_time_settings{
              .default_ = (query.minTransferTime_ == 0 &&
                           query.additionalTransferTime_ == 0 &&
                           query.transferTimeFactor_ == 1.0),
              .min_transfer_time_ = nigiri::duration_t{query.minTransferTime_},
              .additional_time_ =
                  nigiri::duration_t{query.additionalTransferTime_},
              .factor_ = static_cast<float>(query.transferTimeFactor_)},
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

  auto const one = make_place(
      l, time, query.arriveBy_ ? dir_t::kArrival : dir_t::kDeparture);

  auto all = std::vector<api::ReachablePlace>{};
  for (auto i = nigiri::location_idx_t{0U}; i < tt_.n_locations(); ++i) {
    if (state.get_best<0>()[to_idx(i)][0] != unreachable) {
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
          make_place(i, time + std::chrono::minutes{fastest.duration_},
                     query.arriveBy_ ? dir_t::kDeparture : dir_t::kArrival),
          query.arriveBy_ ? -fastest.duration_ : fastest.duration_, fastest.k_);
    }
  }
  return {
      .one_ = std::move(one),
      .all_ = std::move(all),
  };
}

api::Place one_to_all::make_place(nigiri::location_idx_t const l,
                                  nigiri::unixtime_t const t,
                                  dir_t const dir) const {
  auto const pos = tt_.locations_.coordinates_[l];

  if (dir == dir_t::kArrival) {
    return {
        .name_ = std::string{tt_.locations_.names_[l].view()},
        .stopId_ = tags_.id(tt_, l),
        .lat_ = pos.lat(),
        .lon_ = pos.lng(),
        .level_ = static_cast<double>(to_idx(get_lvl(w_, pl_, matches_, l))),
        .arrival_ = t,
    };
  } else {
    return {
        .name_ = std::string{tt_.locations_.names_[l].view()},
        .stopId_ = tags_.id(tt_, l),
        .lat_ = pos.lat(),
        .lon_ = pos.lng(),
        .level_ = static_cast<double>(to_idx(get_lvl(w_, pl_, matches_, l))),
        .departure_ = t,
    };
  }
}

}  // namespace motis::ep
