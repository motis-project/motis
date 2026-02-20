#include "motis/endpoints/one_to_many.h"

#include <chrono>
#include <limits>
#include <optional>

#include "utl/enumerate.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/one_to_all.h"

#include "motis/endpoints/one_to_many_post.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/routing_data.h"
#include "motis/metrics_registry.h"
#include "motis/osr/mode_to_profile.h"
#include "motis/timetable/modes_to_clasz_mask.h"
#include "net/bad_request_exception.h"

namespace motis::ep {

namespace n = nigiri;

constexpr auto kInfinity = std::numeric_limits<double>::infinity();

api::oneToMany_response one_to_many_direct(
    osr::ways const& w,
    osr::lookup const& l,
    api::ModeEnum const mode,
    osr::location const& one,
    std::vector<osr::location> const& many,
    double const max_direct_time,
    double const max_matching_distance,
    osr::direction const dir,
    osr_parameters const& params,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs,
    osr::elevation_storage const* elevations_,
    bool with_distance) {

  utl::verify(mode == api::ModeEnum::BIKE || mode == api::ModeEnum::CAR ||
                  mode == api::ModeEnum::WALK,
              "mode {} not supported for one-to-many",
              boost::json::serialize(boost::json::value_from(mode)));

  auto const profile = to_profile(mode, pedestrian_profile, elevation_costs);
  auto const paths =
      osr::route(to_profile_parameters(profile, params), w, l, profile, one,
                 many, max_direct_time, dir, max_matching_distance, nullptr,
                 nullptr, elevations_, [&](auto&&) { return with_distance; });

  return utl::to_vec(paths, [&](std::optional<osr::path> const& p) {
    return p
        .transform([&](osr::path const& x) {
          return api::Duration{.duration_ = x.cost_,
                               .distance_ = with_distance
                                                ? std::optional{x.dist_}
                                                : std::nullopt};
        })
        .value_or(api::Duration{});
  });
}

template <typename Endpoint, typename Query>
void update_transit_durations(
    api::oneToMany_response& durations,
    Endpoint const& ep,
    Query const& query,
    place_t const& one,
    std::vector<place_t> const& many,
    auto const& time,
    bool const arrive_by,
    std::chrono::seconds const max_travel_time,
    double const max_matching_distance,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs,
    osr_parameters const& osr_params) {
  // TODO Should this always be calculated?
  // Code is similar to One-to-All
  auto const one_modes =
      deduplicate((arrive_by ? query.postTransitModes_ : query.preTransitModes_)
                      .value_or(std::vector{api::ModeEnum::WALK}));
  auto const many_modes =
      deduplicate((arrive_by ? query.preTransitModes_ : query.postTransitModes_)
                      .value_or(std::vector{api::ModeEnum::WALK}));
  auto const max_prepost_seconds = std::min(
      max_travel_time,
      std::chrono::seconds{ep.config_.limits_.value()
                               .street_routing_max_prepost_transit_seconds_});
  auto const one_max_seconds =
      std::min(std::chrono::seconds{(arrive_by ? query.maxPostTransitTime_
                                               : query.maxPreTransitTime_)
                                        .value_or(900)},
               max_prepost_seconds);
  auto const many_max_seconds =
      std::min(std::chrono::seconds{(arrive_by ? query.maxPreTransitTime_
                                               : query.maxPostTransitTime_)
                                        .value_or(900)},
               max_prepost_seconds);
  auto const one_dir =
      arrive_by ? osr::direction::kBackward : osr::direction::kForward;
  auto const unreachable = arrive_by ? n::kInvalidDelta<n::direction::kBackward>
                                     : n::kInvalidDelta<n::direction::kForward>;
  auto const delta_to_seconds = [&](n::delta_t const d) -> double {
    return 60.0 * (arrive_by ? -1 * d : d);
  };
  auto const duration_to_seconds = [](n::duration_t const d) {
    return 60 * d.count();
  };

  auto const r = routing{ep.config_,     ep.w_,   ep.l_,       ep.pl_,
                         ep.elevations_, &ep.tt_, nullptr,     &ep.tags_,
                         ep.loc_tree_,   ep.fa_,  ep.matches_, ep.way_matches_,
                         ep.rt_,         nullptr, ep.gbfs_,    nullptr,
                         nullptr,        nullptr, nullptr,     ep.metrics_};
  auto gbfs_rd = gbfs::gbfs_routing_data{ep.w_, ep.l_, ep.gbfs_};

  auto prepare_stats = std::map<std::string, std::uint64_t>{};

  auto q = n::routing::query{
      .start_time_ = time,
      .start_match_mode_ = get_match_mode(one),
      .start_ = r.get_offsets(nullptr, one, one_dir, one_modes, std::nullopt,
                              std::nullopt, std::nullopt, std::nullopt, false,
                              osr_params, pedestrian_profile, elevation_costs,
                              one_max_seconds, max_matching_distance, gbfs_rd,
                              prepare_stats),
      .td_start_ = r.get_td_offsets(nullptr, nullptr, one, one_dir, one_modes,
                                    osr_params, pedestrian_profile,
                                    elevation_costs, max_matching_distance,
                                    one_max_seconds, time, prepare_stats),
      .max_transfers_ = static_cast<std::uint8_t>(
          query.maxTransfers_.value_or(n::routing::kMaxTransfers)),
      .max_travel_time_ =
          std::chrono::duration_cast<n::duration_t>(max_travel_time),
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
      arrive_by
          ? n::routing::one_to_all<n::direction::kBackward>(ep.tt_, nullptr, q)
          : n::routing::one_to_all<n::direction::kForward>(ep.tt_, nullptr, q);

  auto reachable = n::bitvec{ep.tt_.n_locations()};
  for (auto i = 0U; i != ep.tt_.n_locations(); ++i) {
    if (state.template get_best<0>()[i][0] != unreachable) {
      reachable.set(i);
    }
  }

  for (auto const [i, l] : utl::enumerate(many)) {
    auto best = kInfinity;
    auto const offsets = r.get_offsets(
        nullptr, l,
        arrive_by ? osr::direction::kForward : osr::direction::kBackward,
        many_modes, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        false, osr_params, pedestrian_profile, elevation_costs,
        many_max_seconds, max_matching_distance, gbfs_rd, prepare_stats);

    for (auto const offset : offsets) {
      auto const loc = offset.target();
      if (reachable.test(to_idx(loc))) {
        auto const fastest = n::routing::get_fastest_one_to_all_offsets(
            ep.tt_, state,
            arrive_by ? n::direction::kBackward : n::direction::kForward, loc,
            time, q.max_transfers_);
        auto const total = delta_to_seconds(fastest.duration_) +
                           duration_to_seconds(offset.duration());
        if (total < best) {
          best = total;
        }
      }
    }

    if (best < kInfinity && best <= max_travel_time.count() &&
        (!durations[i].duration_.has_value() ||
         best < *durations[i].duration_)) {
      durations[i].duration_ = best;
    }
  }
}

api::oneToMany_response one_to_many::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::oneToMany_params{url.params()};
  return one_to_many_handle_request(query, w_, l_, elevations_);
}

template <typename Endpoint, typename Query>
api::oneToManyIntermodal_response run_one_to_many_intermodal(
    Endpoint const& ep,
    Query const& query,
    place_t const& one,
    std::vector<place_t> const& many) {
  ep.metrics_->routing_requests_.Increment();

  auto const time = std::chrono::time_point_cast<std::chrono::minutes>(
      *query.time_.value_or(openapi::now()));
  auto const max_travel_time =
      query.maxTravelTime_
          .and_then([](std::int64_t const dur) {
            return std::optional{
                std::chrono::duration_cast<std::chrono::seconds>(
                    n::duration_t{dur})};
          })
          .value_or(kInfinityDuration);
  auto const max_matching_distance = query.maxMatchingDistance_.value_or(250.0);
  auto const arrive_by = query.arriveBy_.value_or(false);

  auto const pedestrian_profile =
      query.pedestrianProfile_.value_or(api::PedestrianProfileEnum::FOOT);
  auto const elevation_costs =
      query.elevationCosts_.value_or(api::ElevationCostsEnum::NONE);
  auto const osr_params = get_osr_parameters(query);

  // Get street routing durations
  utl::verify<net::bad_request_exception>(
      !query.directModes_.has_value() || query.directModes_->size() == 1,
      "Only one direct mode supported. Got {}", query.directModes_->size());
  auto durations =
      query.directModes_
          .transform([&](std::vector<api::ModeEnum> const& direct_modes) {
            auto const to_location = [&](place_t const& p) {
              return get_location(&ep.tt_, ep.w_, ep.pl_, ep.matches_, p);
            };
            return one_to_many_direct(
                *ep.w_, *ep.l_, direct_modes.at(0), to_location(one),
                utl::to_vec(many, to_location),
                static_cast<double>(std::min(
                    {query.maxDirectTime_.value_or(max_travel_time.count()),
                     static_cast<std::int64_t>(max_travel_time.count()),
                     static_cast<std::int64_t>(
                         ep.config_.get_limits()
                             .street_routing_max_direct_seconds_)})),
                max_matching_distance,
                arrive_by ? osr::direction::kBackward
                          : osr::direction::kForward,
                osr_params, pedestrian_profile, elevation_costs, ep.elevations_,
                false);
          })
          .value_or(api::oneToManyIntermodal_response{many.size()});

  update_transit_durations(durations, ep, query, one, many, time, arrive_by,
                           max_travel_time, max_matching_distance,
                           pedestrian_profile, elevation_costs, osr_params);

  return durations;
}

api::oneToManyIntermodal_response one_to_many_intermodal::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::oneToManyIntermodal_params{url.params()};
  auto const one = parse_location(query.one_, ';');
  utl::verify<net::bad_request_exception>(
      one.has_value(), "{} is not a valid geo coordinate", query.one_);

  auto const many = utl::to_vec(query.many_, [](auto&& x) -> place_t {
    auto const y = parse_location(x, ';');
    utl::verify<net::bad_request_exception>(
        y.has_value(), "{} is not a valid geo coordinate", x);
    return *y;
  });
  return run_one_to_many_intermodal(*this, query, *one, many);
}

template api::oneToManyIntermodal_response run_one_to_many_intermodal<
    one_to_many_intermodal,
    api::oneToManyIntermodal_params>(one_to_many_intermodal const&,
                                     api::oneToManyIntermodal_params const&,
                                     place_t const&,
                                     std::vector<place_t> const&);

template api::oneToManyIntermodal_response run_one_to_many_intermodal<
    one_to_many_intermodal_post,
    api::OneToManyIntermodalParams>(one_to_many_intermodal_post const&,
                                    api::OneToManyIntermodalParams const&,
                                    place_t const&,
                                    std::vector<place_t> const&);

}  // namespace motis::ep
