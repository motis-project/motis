#include "motis/endpoints/one_to_many.h"

#include <chrono>
#include <limits>
#include <optional>

#include "utl/enumerate.h"

#include "net/too_many_exception.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/routing/one_to_all.h"

#include "motis/config.h"
#include "motis/endpoints/one_to_many_post.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/routing_data.h"
#include "motis/osr/mode_to_profile.h"
#include "motis/timetable/modes_to_clasz_mask.h"

namespace motis::ep {

namespace n = nigiri;

constexpr auto const kInfinity = std::numeric_limits<double>::infinity();

api::oneToMany_response one_to_many_direct(
    config const& config,
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
    bool const with_distance) {
  auto const max_many = config.get_limits().onetomany_max_many_;
  auto const max_direct_time_limit =
      config.get_limits().street_routing_max_direct_seconds_;

  utl::verify<net::too_many_exception>(
      many.size() <= max_many,
      "number of many locations too high ({} > {}). The server admin can "
      "change this limit in config.yml with 'onetomany_max_many'. "
      "See documentation for details.",
      many.size(), max_many);
  utl::verify<net::too_many_exception>(
      max_direct_time <= max_direct_time_limit,
      "maximun travel time too high ({} > {}). The server admin can "
      "change this limit in config.yml with "
      "'street_routing_max_direct_seconds'. "
      "See documentation for details.",
      max_direct_time, max_direct_time_limit);
  utl::verify<net::bad_request_exception>(
      mode == api::ModeEnum::BIKE || mode == api::ModeEnum::CAR ||
          mode == api::ModeEnum::WALK,
      "mode {} not supported for one-to-many", fmt::streamed(mode));

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

double duration_to_seconds(n::duration_t const d) { return 60 * d.count(); }

template <typename Endpoint, typename Query>
api::oneToManyIntermodal_response add_transit_durations(
    api::oneToMany_response&& direct_durations,
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
  // Code is similar to One-to-All
  auto const one_modes =
      deduplicate(arrive_by ? query.postTransitModes_ : query.preTransitModes_);
  auto const many_modes =
      deduplicate(arrive_by ? query.preTransitModes_ : query.postTransitModes_);
  auto const max_prepost_seconds = std::min(
      max_travel_time,
      std::chrono::seconds{ep.config_.limits_.value()
                               .street_routing_max_prepost_transit_seconds_});
  auto const one_max_seconds =
      std::min(std::chrono::seconds{arrive_by ? query.maxPostTransitTime_
                                              : query.maxPreTransitTime_},
               max_prepost_seconds);
  auto const many_max_seconds =
      std::min(std::chrono::seconds{arrive_by ? query.maxPreTransitTime_
                                              : query.maxPostTransitTime_},
               max_prepost_seconds);
  auto const one_dir =
      arrive_by ? osr::direction::kBackward : osr::direction::kForward;
  auto const unreachable = arrive_by ? n::kInvalidDelta<n::direction::kBackward>
                                     : n::kInvalidDelta<n::direction::kForward>;

  auto const r = routing{ep.config_,     ep.w_,   ep.l_,       ep.pl_,
                         ep.elevations_, &ep.tt_, nullptr,     &ep.tags_,
                         ep.loc_tree_,   ep.fa_,  ep.matches_, ep.way_matches_,
                         ep.rt_,         nullptr, ep.gbfs_,    nullptr,
                         nullptr,        nullptr, nullptr,     ep.metrics_};
  auto gbfs_rd = gbfs::gbfs_routing_data{ep.w_, ep.l_, ep.gbfs_};

  auto prepare_stats = std::map<std::string, std::uint64_t>{};

  auto q = n::routing::query{
      .start_time_ = time,
      .start_match_mode_ = get_match_mode(r, one),
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
      .prf_idx_ = query.useRoutedTransfers_
                      ? (query.pedestrianProfile_ ==
                                 api::PedestrianProfileEnum::WHEELCHAIR
                             ? n::kWheelchairProfile
                             : n::kFootProfile)
                      : n::kDefaultProfile,
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

  auto pareto_sets = std::vector<api::ParetoSet>{};

  auto const dir = arrive_by ? n::direction::kBackward : n::direction::kForward;
  auto totals = n::vector<double>{};
  for (auto const [i, l] : utl::enumerate(many)) {
    totals.clear();
    totals.resize(q.max_transfers_, kInfinity);
    auto const offsets = r.get_offsets(
        nullptr, l,
        arrive_by ? osr::direction::kForward : osr::direction::kBackward,
        many_modes, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        false, osr_params, pedestrian_profile, elevation_costs,
        many_max_seconds, max_matching_distance, gbfs_rd, prepare_stats);
    for (auto const offset : offsets) {
      auto const loc = offset.target();
      if (reachable.test(to_idx(loc))) {
        auto const base = duration_to_seconds(offset.duration());
        n::routing::for_each_one_to_all_round_time(
            ep.tt_, state, dir, loc, time, q.max_transfers_,
            [&](std::uint8_t const k, n::duration_t const d) {
              if (k != std::uint8_t{0U}) {
                auto const total = base + duration_to_seconds(d);
                totals[k - 1U] = std::min(totals[k - 1], total);
              }
            });
      }
    }
    auto durations = std::vector<api::Duration>{};
    if (direct_durations[i].duration_.has_value()) {
      direct_durations[i].k_ = 0U;
      durations.push_back(std::move(direct_durations[i]));
    }
    auto best = kInfinity;
    for (auto const [j, d] : utl::enumerate(totals)) {
      if (d < kInfinity && d < best) {
        durations.emplace_back(d, j + 1);
        best = d;
      }
    }
    pareto_sets.emplace_back(std::move(durations));
  }
  return pareto_sets;
}

api::oneToMany_response one_to_many::operator()(
    boost::urls::url_view const& url) const {
  auto const query = api::oneToMany_params{url.params()};
  return one_to_many_handle_request(config_, query, w_, l_, elevations_);
}

template <typename Endpoint, typename Query>
api::oneToManyIntermodal_response run_one_to_many_intermodal(
    Endpoint const& ep,
    Query const& query,
    place_t const& one,
    std::vector<place_t> const& many) {
  auto const time = std::chrono::time_point_cast<std::chrono::minutes>(
      *query.time_.value_or(openapi::now()));

  auto const max_travel_time =
      query.maxTravelTime_
          .transform([](std::int64_t const dur) {
            using namespace std::chrono;
            return duration_cast<seconds>(minutes{dur});
          })
          .value_or(n::routing::kMaxTravelTime);

  auto const osr_params = get_osr_parameters(query);

  // Get street routing durations
  auto const to_location = [&](place_t const& p) {
    return get_location(&ep.tt_, ep.w_, ep.pl_, ep.matches_, p);
  };
  auto durations = one_to_many_direct(
      ep.config_, *ep.w_, *ep.l_, query.directMode_, to_location(one),
      utl::to_vec(many, to_location),
      static_cast<double>(std::min(
          {std::max({query.maxDirectTime_, query.maxPreTransitTime_,
                     query.maxPostTransitTime_}),
           static_cast<std::int64_t>(max_travel_time.count()),
           static_cast<std::int64_t>(
               ep.config_.get_limits().street_routing_max_direct_seconds_)})),
      query.maxMatchingDistance_,
      query.arriveBy_ ? osr::direction::kBackward : osr::direction::kForward,
      osr_params, query.pedestrianProfile_, query.elevationCosts_,
      ep.elevations_, query.withDistance_);

  return add_transit_durations(
      std::move(durations), ep, query, one, many, time, query.arriveBy_,
      max_travel_time, query.maxMatchingDistance_, query.pedestrianProfile_,
      query.elevationCosts_, osr_params);
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

template api::oneToManyIntermodalPost_response run_one_to_many_intermodal<
    one_to_many_intermodal_post,
    api::OneToManyIntermodalParams>(one_to_many_intermodal_post const&,
                                    api::OneToManyIntermodalParams const&,
                                    place_t const&,
                                    std::vector<place_t> const&);

}  // namespace motis::ep
