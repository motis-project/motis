#include "motis/endpoints/one_to_many.h"

#include <limits>
#include <optional>

#include "utl/enumerate.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/one_to_all.h"
#include "nigiri/types.h"

#include "motis/endpoints/one_to_many_post.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/routing_data.h"
#include "motis/metrics_registry.h"
#include "motis/osr/mode_to_profile.h"
#include "motis/timetable/modes_to_clasz_mask.h"

namespace motis::ep {

namespace n = nigiri;

api::oneToMany_response one_to_many_direct(
    osr::ways const& w_,
    osr::lookup const& l_,
    api::ModeEnum const mode,
    osr::location const& one,
    std::vector<osr::location> const& many,
    double const max_travel_time,
    double const max_matching_distance,
    bool const arrive_by,
    // osr::direction const dir,
    // osr::search_profile const profile,
    osr_parameters const& params,
    // osr::profile_parameters const& params,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs,
    osr::elevation_storage const* elevations_) {

  utl::verify(mode == api::ModeEnum::BIKE || mode == api::ModeEnum::CAR ||
                  mode == api::ModeEnum::WALK,
              "mode {} not supported for one-to-many",
              boost::json::serialize(boost::json::value_from(mode)));

  auto const profile = to_profile(mode, pedestrian_profile, elevation_costs);
  auto const paths = osr::route(
      to_profile_parameters(profile, params), w_, l_, profile, one, many,
      max_travel_time,
      arrive_by ? osr::direction::kBackward : osr::direction::kForward,
      max_matching_distance, nullptr, nullptr, elevations_);

  return utl::to_vec(paths, [](std::optional<osr::path> const& p) {
    return p.has_value() ? api::Duration{.duration_ = p->cost_}
                         : api::Duration{};
  });
}
api::oneToMany_response one_to_many::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::oneToMany_params{url.params()};
  return one_to_many_handle_request(query, w_, l_, elevations_);
}

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

  auto const pedestrian_profile =
      query.pedestrianProfile_.value_or(api::PedestrianProfileEnum::FOOT);
  auto const elevation_costs =
      query.elevationCosts_.value_or(api::ElevationCostsEnum::NONE);

  // Get street routing durations
  utl::verify(
      !query.directModes_.has_value() || query.directModes_->size() == 1,
      "Only one direct mode supported. Got {}", query.directModes_->size());
  auto durations =
      query.directModes_
          ? one_to_many_direct(
                *ep.w_, *ep.l_, (*query.directModes_)[0], *one, many,
                std::min(query.maxDirectTime_.value_or(query.maxTravelTime_),
                         query.maxTravelTime_),
                query.maxMatchingDistance_, query.arriveBy_,
                get_osr_parameters(query), pedestrian_profile, elevation_costs,
                ep.elevations_)
          : api::oneToManyIntermodal_response{many.size()};

  // TODO Should this always be calculated?
  // TODO What if transitModes.empty() and maxDirectTime == 0?
  // Following code is similar to One-to-All
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
  constexpr auto const kInf = std::numeric_limits<double>::infinity();

  auto const to_duration = [&](n::delta_t const d) -> double {
    return 60.0 * (query.arriveBy_ ? -1 * d : d);
  };
  auto const to_seconds = [](n::duration_t const d) { return 60 * d.count(); };
  auto const unreachable = query.arriveBy_
                               ? nigiri::kInvalidDelta<n::direction::kBackward>
                               : nigiri::kInvalidDelta<n::direction::kForward>;

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

  // Compute and update durations using transits

  auto const state =
      query.arriveBy_
          ? n::routing::one_to_all<n::direction::kBackward>(
                ep.tt_, nullptr, q)  // Missing RT support
          : n::routing::one_to_all<n::direction::kForward>(ep.tt_, nullptr, q);
  auto reachable = nigiri::bitvec{ep.tt_.n_locations()};
  for (auto i = 0U; i != ep.tt_.n_locations(); ++i) {
    if (state.template get_best<0>()[i][0] != unreachable) {
      reachable.set(i);
    }
  }
  for (auto const [i, l] : utl::enumerate(many)) {
    auto best = kInf;
    auto const offsets = r.get_offsets(
        nullptr, l, many_dir, many_modes, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, false, osr_params, pedestrian_profile,
        elevation_costs, many_max_time, query.maxMatchingDistance_, gbfs_rd,
        prepare_stats);
    for (auto const offset : offsets) {
      auto const loc = offset.target();
      if (reachable.test(to_idx(loc))) {
        auto const fastest = n::routing::get_fastest_one_to_all_offsets(
            ep.tt_, state,
            query.arriveBy_ ? n::direction::kBackward : n::direction::kForward,
            loc, time, q.max_transfers_);
        auto const total =
            to_duration(fastest.duration_) + to_seconds(offset.duration());
        if (total < best) {
          best = total;
        }
      }
    }
    if (best < kInf && (!durations[i].duration_.has_value() ||
                        best < *durations[i].duration_)) {
      durations[i].duration_ = best;
    }
  }

  return durations;
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
