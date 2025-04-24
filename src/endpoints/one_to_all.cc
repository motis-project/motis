#include "motis/endpoints/one_to_all.h"

#include <chrono>
#include <vector>

#include "utl/verify.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/one_to_all.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/routing_data.h"
#include "motis/place.h"
#include "motis/timetable/modes_to_clasz_mask.h"

namespace motis::ep {

namespace n = nigiri;

constexpr auto const kMaxResults = 65535U;
constexpr auto const kMaxTravelMinutes = 90U;

api::Reachable one_to_all::operator()(boost::urls::url_view const& url) const {
  auto const query = api::oneToAll_params{url.params()};
  utl::verify(query.maxTravelTime_ <= kMaxTravelMinutes,
              "maxTravelTime too large: {} > {}", query.maxTravelTime_,
              kMaxTravelMinutes);
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

  auto const make_place = [&](place_t const& p, n::unixtime_t const t,
                              n::event_type const ev) {
    auto place = to_place(&tt_, &tags_, w_, pl_, matches_, p);
    if (ev == n::event_type::kArr) {
      place.arrival_ = t;
    } else {
      place.departure_ = t;
    }
    return place;
  };

  auto const time = std::chrono::time_point_cast<std::chrono::minutes>(
      *query.time_.value_or(openapi::now()));
  auto const max_travel_time = n::duration_t{query.maxTravelTime_};
  auto const one = get_place(&tt_, &tags_, query.one_);
  auto const one_modes = deduplicate(query.arriveBy_ ? query.postTransitModes_
                                                     : query.preTransitModes_);
  auto const one_max_time = std::min(
      std::chrono::seconds{query.arriveBy_ ? query.maxPostTransitTime_
                                           : query.maxPreTransitTime_},
      std::chrono::duration_cast<std::chrono::seconds>(max_travel_time));
  auto const one_dir =
      query.arriveBy_ ? osr::direction::kBackward : osr::direction::kForward;

  auto const r =
      routing{config_,   w_,       l_,  pl_,     elevations_, &tt_,   &tags_,
              loc_tree_, matches_, rt_, nullptr, gbfs_,       nullptr};
  auto gbfs_rd = gbfs::gbfs_routing_data{w_, l_, gbfs_};

  auto const q = n::routing::query{
      .start_time_ = time,
      .start_match_mode_ = get_match_mode(one),
      .start_ = r.get_offsets(
          one, one_dir, one_modes, std::nullopt, std::nullopt, std::nullopt,
          query.pedestrianProfile_, query.elevationCosts_, one_max_time,
          query.maxMatchingDistance_, gbfs_rd),
      .td_start_ = r.get_td_offsets(
          rt_->e_.get(), one, one_dir, one_modes, query.pedestrianProfile_,
          query.elevationCosts_, query.maxMatchingDistance_, one_max_time),
      .max_transfers_ = static_cast<std::uint8_t>(
          query.maxTransfers_.value_or(n::routing::kMaxTransfers)),
      .max_travel_time_ = max_travel_time,
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

  auto reachable = nigiri::bitvec{tt_.n_locations()};
  for (auto i = 0U; i != tt_.n_locations(); ++i) {
    if (state.get_best<0>()[i][0] != unreachable) {
      reachable.set(i);
    }
  }
  utl::verify(reachable.count() <= kMaxResults, "too many results: {} > {}",
              reachable.count(), kMaxResults);

  auto all = std::vector<api::ReachablePlace>{};
  all.reserve(reachable.count());
  auto const all_ev =
      query.arriveBy_ ? n::event_type::kDep : n::event_type::kArr;
  reachable.for_each_set_bit([&](auto const i) {
    auto const l = n::location_idx_t{i};
    auto const fastest = n::routing::get_fastest_one_to_all_offsets(
        tt_, state,
        query.arriveBy_ ? n::direction::kBackward : n::direction::kForward, l,
        time, q.max_transfers_);

    all.push_back(api::ReachablePlace{
        make_place(tt_location{l},
                   time + std::chrono::minutes{fastest.duration_}, all_ev),
        query.arriveBy_ ? -fastest.duration_ : fastest.duration_, fastest.k_});
  });

  return {
      .one_ = make_place(
          one, time,
          query.arriveBy_ ? n::event_type::kArr : n::event_type::kDep),
      .all_ = std::move(all),
  };
}

}  // namespace motis::ep
