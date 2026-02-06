#include "motis/endpoints/one_to_many.h"

#include <optional>

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/one_to_all.h"

#include "osr/location.h"

#include "motis/endpoints/one_to_many_post.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/routing_data.h"
#include "motis/metrics_registry.h"
#include "motis/timetable/modes_to_clasz_mask.h"

namespace motis::ep {

namespace n = nigiri;

api::oneToMany_response one_to_many::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::oneToMany_params{url.params()};
  return one_to_many_handle_request(query, w_, l_, elevations_);
}

/*
  std::string one_{};
  std::vector<std::string> many_{};
  std::optional<openapi::date_time_t> time_{};
  std::int64_t maxTravelTime_{};
  double maxMatchingDistance_{};
  bool arriveBy_{};
  std::optional<std::int64_t> maxTransfers_{};
  std::int64_t minTransferTime_{0};
  std::int64_t additionalTransferTime_{0};
  double transferTimeFactor_{1.0};
  bool useRoutedTransfers_{false};
  PedestrianProfileEnum pedestrianProfile_{PedestrianProfileEnum::FOOT};
  std::optional<PedestrianSpeed> pedestrianSpeed_{};
  std::optional<CyclingSpeed> cyclingSpeed_{};
  ElevationCostsEnum elevationCosts_{ElevationCostsEnum::NONE};
  std::vector<ModeEnum> transitModes_{std::vector<ModeEnum>{ModeEnum::TRANSIT}};
  std::vector<ModeEnum> preTransitModes_{std::vector<ModeEnum>{ModeEnum::WALK}};
  std::vector<ModeEnum>
  postTransitModes_{std::vector<ModeEnum>{ModeEnum::WALK}};
  std::vector<ModeEnum> directModes_{std::vector<ModeEnum>{ModeEnum::WALK}};
  std::int64_t maxPreTransitTime_{900};
  std::int64_t maxPostTransitTime_{900};
  std::int64_t maxDirectTime_{1800};
  bool requireBikeTransport_{false};
  bool requireCarTransport_{false};
*/

// TODO Add direct routing

template <typename Endpoint, typename Query>
api::oneToManyIntermodal_response run_one_to_many_intermodal(
    Endpoint const& ep, Query const& query) {
  ep.metrics_->routing_requests_.Increment();

  auto const time = std::chrono::time_point_cast<std::chrono::minutes>(
      *query.time_.value_or(openapi::now()));
  auto const max_travel_time = n::duration_t{query.maxTravelTime_};

  auto const one = parse_location(query.one_, ';');
  utl::verify(one.has_value(), "{} is not a valid geo coordinate", query.one_);

  auto const many = utl::to_vec(query.many_, [](auto&& x) {
    auto const y = parse_location(x, ';');
    utl::verify(y.has_value(), "{} is not a valid geo coordinate", x);
    return *y;
  });

  auto const one_modes = deduplicate(
      (query.arriveBy_ ? query.postTransitModes_ : query.preTransitModes_)
          .value_or(std::vector{api::ModeEnum::WALK}));
  auto const many_modes = deduplicate(
      (query.arriveBy_ ? query.preTransitModes_ : query.postTransitModes_)
          .value_or(std::vector{api::ModeEnum::WALK}));
  auto const max_travel_time_limit = std::min(
      std::chrono::duration_cast<std::chrono::seconds>(max_travel_time),
      std::chrono::seconds{ep.config_.limits_.value()
                               .street_routing_max_prepost_transit_seconds_});
  auto const one_max_time =
      std::min(std::chrono::seconds{(query.arriveBy_ ? query.maxPostTransitTime_
                                                     : query.maxPreTransitTime_)
                                        .value_or(900)},
               max_travel_time_limit);
  auto const many_max_time = std::min(
      std::chrono::seconds{(query.arriveBy_ ? query.maxPreTransitTime_
                                            : query.maxPostTransitTime_)
                               .value_or(900)},
      max_travel_time_limit);
  auto const one_dir =
      query.arriveBy_ ? osr::direction::kBackward : osr::direction::kForward;
  auto const many_dir =
      query.arriveBy_ ? osr::direction::kForward : osr::direction::kBackward;
  auto const pedestrian_profile =
      query.pedestrianProfile_.value_or(api::PedestrianProfileEnum::FOOT);
  auto const elevation_costs =
      query.elevationCosts_.value_or(api::ElevationCostsEnum::NONE);

  auto const r = routing{ep.config_,     ep.w_,   ep.l_,       ep.pl_,
                         ep.elevations_, &ep.tt_, nullptr,     &ep.tags_,
                         ep.loc_tree_,   ep.fa_,  ep.matches_, ep.way_matches_,
                         ep.rt_,         nullptr, ep.gbfs_,    nullptr,
                         nullptr,        nullptr, nullptr,     ep.metrics_};
  auto gbfs_rd = gbfs::gbfs_routing_data{ep.w_, ep.l_, ep.gbfs_};

  auto const osr_params = get_osr_parameters(query);
  auto prepare_stats = std::map<std::string, std::uint64_t>{};

  auto q = n::routing::query{
      .start_time_ = time,
      .start_match_mode_ = get_match_mode(*one),
      .start_ = r.get_offsets(nullptr, *one, one_dir, one_modes, std::nullopt,
                              std::nullopt, std::nullopt, std::nullopt, false,
                              osr_params, pedestrian_profile, elevation_costs,
                              one_max_time, query.maxMatchingDistance_, gbfs_rd,
                              prepare_stats),
      .td_start_ = r.get_td_offsets(nullptr, nullptr, *one, one_dir, one_modes,
                                    osr_params, pedestrian_profile,
                                    elevation_costs, query.maxMatchingDistance_,
                                    one_max_time, time, prepare_stats),
      .max_transfers_ = static_cast<std::uint8_t>(
          query.maxTransfers_.value_or(n::routing::kMaxTransfers)),
      .max_travel_time_ = max_travel_time,
      .prf_idx_ = query.useRoutedTransfers_.value_or(false)
                      ? (query.pedestrianProfile_ ==
                                 api::PedestrianProfileEnum::WHEELCHAIR
                             ? n::kWheelchairProfile
                             : n::kFootProfile)
                      : n::kDefaultProfile,
      .allowed_claszes_ = to_clasz_mask(
          query.transitModes_.value_or(std::vector{api::ModeEnum::TRANSIT})),
      .require_bike_transport_ = query.requireBikeTransport_.value_or(false),
      .require_car_transport_ = query.requireCarTransport_.value_or(false),
      .transfer_time_settings_ =
          n::routing::transfer_time_settings{
              .default_ = (query.minTransferTime_ == 0 &&
                           query.additionalTransferTime_ == 0 &&
                           query.transferTimeFactor_ == 1.0),
              .min_transfer_time_ =
                  n::duration_t{query.minTransferTime_.value_or(0)},
              .additional_time_ =
                  n::duration_t{query.additionalTransferTime_.value_or(0)},
              .factor_ =
                  static_cast<float>(query.transferTimeFactor_.value_or(1.0))},
  };

  if (ep.tt_.locations_.footpaths_out_.at(q.prf_idx_).empty()) {
    q.prf_idx_ = n::kDefaultProfile;
  }
  // Up to now same as one_to_many_im
  auto const state =
      query.arriveBy_
          ? n::routing::one_to_all<n::direction::kBackward>(
                ep.tt_, nullptr, q)  // Missing RT support
          : n::routing::one_to_all<n::direction::kForward>(ep.tt_, nullptr, q);

  auto const unreachable = query.arriveBy_
                               ? nigiri::kInvalidDelta<n::direction::kBackward>
                               : nigiri::kInvalidDelta<n::direction::kForward>;
  auto reachable = nigiri::bitvec{ep.tt_.n_locations()};
  for (auto i = 0U; i != ep.tt_.n_locations(); ++i) {
    if (state.template get_best<0>()[i][0] != unreachable) {
      reachable.set(i);
    }
  }
  auto const dir =
      query.arriveBy_ ? n::direction::kBackward : n::direction::kForward;

  return utl::to_vec(
      // many_offsets,
      // [&](std::vector<n::routing::offset> const& offsets) -> api::Duration {
      many, [&](osr::location const l) -> api::Duration {
        auto const offsets = r.get_offsets(
            nullptr, l, many_dir, many_modes, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, false, osr_params, pedestrian_profile,
            elevation_costs, many_max_time, query.maxMatchingDistance_, gbfs_rd,
            prepare_stats);
        // fmt::println("Testing {} offsets ...", offsets.size());
        auto best = unreachable;
        for (auto const offset : offsets) {
          auto const loc = offset.target();
          if (reachable.test(to_idx(loc))) {
            auto const fastest = n::routing::get_fastest_one_to_all_offsets(
                ep.tt_, state, dir, loc, time, q.max_transfers_);
            auto const total = static_cast<n::delta_t>(
                fastest.duration_ + offset.duration().count());
            // fmt::println(
            //     "Testing {}: fastest={}, off_dur={}, total={} "
            //     "(best={}) (ur={}) (k={}) (pos={})",
            //     loc, fastest.duration_, offset.duration().count(), total,
            //     best, unreachable, fastest.k_,
            //     ep.tt_.locations_.coordinates_[loc]);
            if (total < best) {
              best = total;
            }
          }  // else
             // fmt::println("Target {} is unreachable (dur={}) (pos={})", loc,
             //              offset.duration(),
             //              ep.tt_.locations_.coordinates_[loc]);
        }
        return best < unreachable ? api::Duration{best} : api::Duration{};
      });
}

api::oneToManyIntermodal_response one_to_many_intermodal::operator()(
    boost::urls::url_view const& url) const {
  fmt::println("GET(1)");
  auto const query = api::oneToManyIntermodal_params{url.params()};
  return run_one_to_many_intermodal(*this, query);
}

template api::oneToManyIntermodal_response
run_one_to_many_intermodal<one_to_many_intermodal,
                           api::oneToManyIntermodal_params>(
    one_to_many_intermodal const& ep,
    api::oneToManyIntermodal_params const& query);

template api::oneToManyIntermodal_response
run_one_to_many_intermodal<one_to_many_intermodal_post,
                           api::OneToManyIntermodalParams>(
    one_to_many_intermodal_post const& ep,
    api::OneToManyIntermodalParams const& query);

}  // namespace motis::ep
