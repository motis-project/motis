#include "motis/endpoints/one_to_all.h"

#include <chrono>
#include <vector>

#include "utl/verify.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/location_match_mode.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/one_to_all.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/place.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/modes_to_clasz_mask.h"

namespace motis::ep {

namespace n = nigiri;

api::Reachable one_to_all::operator()(boost::urls::url_view const& url) const {
  auto const query = api::oneToAll_params{url.params()};
  if (query.maxTransfers_.has_value()) {
    utl::verify(query.maxTransfers_ >= 0U, "maxTransfers < 0: {}",
                *query.maxTransfers_);
    utl::verify(query.maxTransfers_ <= n::routing::kMaxTransfers,
                "maxTransfers > {}: {}", n::routing::kMaxTransfers,
                *query.maxTransfers_);
  }

  auto const unreachable = query.arriveBy_
                               ? n::kInvalidDelta<n::direction::kBackward>
                               : n::kInvalidDelta<n::direction::kForward>;
  auto const rtt = rt_->rtt_.get();

  auto const make_place = [&](n::location_idx_t const l, n::unixtime_t const t,
                              n::event_type const ev) {
    auto place = to_place(&tt_, &tags_, w_, pl_, matches_, tt_location{l});
    if (ev == n::event_type::kArr) {
      place.arrival_ = t;
    } else {
      place.departure_ = t;
    }
    return place;
  };

  auto const time = std::chrono::time_point_cast<std::chrono::minutes>(
      *query.time_.value_or(openapi::now()));
  auto const l = tags_.get_location(tt_, query.one_);

  auto const q = n::routing::query{
      .start_time_ = time,
      .start_match_mode_ = n::routing::location_match_mode::kEquivalent,
      .start_ = {{l, n::duration_t{}, n::transport_mode_id_t{0U}}},
      .max_transfers_ = static_cast<std::uint8_t>(
          query.maxTransfers_.value_or(n::routing::kMaxTransfers)),
      .max_travel_time_ = n::duration_t{query.maxTravelTime_},
      .prf_idx_ = static_cast<n::profile_idx_t>(
          query.useRoutedTransfers_
              ? (query.pedestrianProfile_ ==
                         api::PedestrianProfileEnum::WHEELCHAIR
                     ? 2U
                     : 1U)
              : 0U),
      .allowed_claszes_ = to_clasz_mask(query.transitModes_),
      .require_bike_transport_ = query.requireBikeTransport_,
      .transfer_time_settings_ =
          n::routing::transfer_time_settings{
              .default_ = (query.minTransferTime_ == 0 &&
                           query.additionalTransferTime_ == 0 &&
                           query.transferTimeFactor_ == 1.0),
              .min_transfer_time_ = n::duration_t{query.minTransferTime_},
              .additional_time_ = n::duration_t{query.additionalTransferTime_},
              .factor_ = static_cast<float>(query.transferTimeFactor_)},
  };

  auto const state =
      query.arriveBy_
          ? n::routing::one_to_all<n::direction::kBackward>(tt_, rtt, q)
          : n::routing::one_to_all<n::direction::kForward>(tt_, rtt, q);

  auto all = std::vector<api::ReachablePlace>{};
  auto const all_ev =
      query.arriveBy_ ? n::event_type::kDep : n::event_type::kArr;
  for (auto i = n::location_idx_t{0U}; i != tt_.n_locations(); ++i) {
    if (state.get_best<0>()[to_idx(i)][0] == unreachable) {
      continue;
    }

    auto const fastest = n::routing::get_fastest_one_to_all_offsets(
        tt_, state,
        query.arriveBy_ ? n::direction::kBackward : n::direction::kForward, i,
        time, q.max_transfers_);

    all.push_back(api::ReachablePlace{
        make_place(i, time + std::chrono::minutes{fastest.duration_}, all_ev),
        query.arriveBy_ ? -fastest.duration_ : fastest.duration_, fastest.k_});
  }

  return {
      .one_ = make_place(
          l, time, query.arriveBy_ ? n::event_type::kArr : n::event_type::kDep),
      .all_ = std::move(all),
  };
}

}  // namespace motis::ep
