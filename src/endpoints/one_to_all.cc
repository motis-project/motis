#include "motis/endpoints/one_to_all.h"

#include <chrono>
#include <vector>

#include "utl/verify.h"

#include "net/bad_request_exception.h"
#include "net/too_many_exception.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/one_to_all.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/routing_data.h"
#include "motis/metrics_registry.h"
#include "motis/place.h"
#include "motis/timetable/modes_to_clasz_mask.h"

namespace motis::ep {

namespace n = nigiri;

api::Reachable one_to_all::operator()(boost::urls::url_view const& url) const {
  metrics_->routing_requests_.Increment();

  auto const max_travel_minutes =
      config_.limits_.value().onetoall_max_travel_minutes_;
  auto const query = api::oneToAll_params{url.params()};
  utl::verify<net::too_many_exception>(
      query.maxTravelTime_ <= max_travel_minutes,
      "maxTravelTime too large ({} > {}). The server admin can change "
      "this limit in config.yml with 'onetoall_max_travel_minutes'. "
      "See documentation for details.",
      query.maxTravelTime_, max_travel_minutes);
  if (query.maxTransfers_.has_value()) {
    utl::verify<net::bad_request_exception>(query.maxTransfers_ >= 0U,
                                            "maxTransfers < 0: {}",
                                            *query.maxTransfers_);
    utl::verify<net::too_many_exception>(
        query.maxTransfers_ <= n::routing::kMaxTransfers,
        "maxTransfers > {}: {}", n::routing::kMaxTransfers,
        *query.maxTransfers_);
  }

  auto const unreachable = query.arriveBy_
                               ? n::kInvalidDelta<n::direction::kBackward>
                               : n::kInvalidDelta<n::direction::kForward>;

  auto const make_place = [&](place_t const& p, n::unixtime_t const t,
                              n::event_type const ev) {
    auto place = to_place(&tt_, &tags_, w_, pl_, matches_, ae_, tz_, {}, p);
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
      std::min(
          std::chrono::duration_cast<std::chrono::seconds>(max_travel_time),
          std::chrono::seconds{
              config_.limits_.value()
                  .street_routing_max_prepost_transit_seconds_}));
  auto const one_dir =
      query.arriveBy_ ? osr::direction::kBackward : osr::direction::kForward;

  auto const r = routing{
      config_, w_,        l_,      pl_,      elevations_,  &tt_,    nullptr,
      &tags_,  loc_tree_, fa_,     matches_, way_matches_, rt_,     nullptr,
      gbfs_,   nullptr,   nullptr, nullptr,  nullptr,      metrics_};
  auto gbfs_rd = gbfs::gbfs_routing_data{w_, l_, gbfs_};

  auto const osr_params = get_osr_parameters(query);
  auto prepare_stats = std::map<std::string, std::uint64_t>{};
  auto q = n::routing::query{
      .start_time_ = time,
      .start_match_mode_ = get_match_mode(one),
      .start_ = r.get_offsets(
          nullptr, one, one_dir, one_modes, std::nullopt, std::nullopt,
          std::nullopt, std::nullopt, false, osr_params,
          query.pedestrianProfile_, query.elevationCosts_, one_max_time,
          query.maxMatchingDistance_, gbfs_rd, prepare_stats),
      .td_start_ = r.get_td_offsets(
          nullptr, nullptr, one, one_dir, one_modes, osr_params,
          query.pedestrianProfile_, query.elevationCosts_,
          query.maxMatchingDistance_, one_max_time, time, prepare_stats),
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
      .require_car_transport_ = query.requireCarTransport_,
      .transfer_time_settings_ =
          n::routing::transfer_time_settings{
              .default_ = (query.minTransferTime_ == 0 &&
                           query.additionalTransferTime_ == 0 &&
                           query.transferTimeFactor_ == 1.0),
              .min_transfer_time_ = n::duration_t{query.minTransferTime_},
              .additional_time_ = n::duration_t{query.additionalTransferTime_},
              .factor_ = static_cast<float>(query.transferTimeFactor_)},
  };

  if (tt_.locations_.footpaths_out_.at(q.prf_idx_).empty()) {
    q.prf_idx_ = 0U;
  }

  auto const state =
      query.arriveBy_
          ? n::routing::one_to_all<n::direction::kBackward>(tt_, nullptr, q)
          : n::routing::one_to_all<n::direction::kForward>(tt_, nullptr, q);

  auto reachable = nigiri::bitvec{tt_.n_locations()};
  for (auto i = 0U; i != tt_.n_locations(); ++i) {
    if (state.get_best<0>()[i][0] != unreachable) {
      reachable.set(i);
    }
  }

  auto const max_results = config_.limits_.value().onetoall_max_results_;
  utl::verify<net::too_many_exception>(reachable.count() <= max_results,
                                       "too many results: {} > {}",
                                       reachable.count(), max_results);

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
