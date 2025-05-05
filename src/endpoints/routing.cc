#include "motis/endpoints/routing.h"

#include <algorithm>

#include "boost/thread/tss.hpp"

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"

#include "osr/platforms.h"
#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"
#include "osr/routing/sharing_data.h"

#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/special_stations.h"

#include "motis/constants.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/data.h"
#include "motis/gbfs/mode.h"
#include "motis/gbfs/osr_profile.h"
#include "motis/gbfs/routing_data.h"
#include "motis/journey_to_response.h"
#include "motis/max_distance.h"
#include "motis/mode_to_profile.h"
#include "motis/odm/meta_router.h"
#include "motis/parse_location.h"
#include "motis/street_routing.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/modes_to_clasz_mask.h"
#include "motis/timetable/time_conv.h"
#include "motis/update_rtt_td_footpaths.h"

namespace n = nigiri;
using namespace std::chrono_literals;

namespace motis::ep {

using td_offsets_t =
    n::hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<n::routing::search_state> search_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<n::routing::raptor_state> raptor_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<osr::bitvec<osr::node_idx_t>> blocked;

place_t get_place(n::timetable const* tt,
                  tag_lookup const* tags,
                  std::string_view input) {
  if (auto const location = parse_location(input); location.has_value()) {
    return *location;
  }
  utl::verify(tt != nullptr && tags != nullptr,
              R"(could not parse location (no timetable loaded): "{}")", input);
  return tt_location{tags->get_location(*tt, input)};
}

bool is_intermodal(place_t const& p) {
  return std::holds_alternative<osr::location>(p);
}

n::routing::location_match_mode get_match_mode(place_t const& p) {
  return is_intermodal(p) ? n::routing::location_match_mode::kIntermodal
                          : n::routing::location_match_mode::kEquivalent;
}

std::vector<n::routing::offset> station_start(n::location_idx_t const l) {
  return {{l, n::duration_t{0U}, 0U}};
}

td_offsets_t get_td_offsets(routing const& r,
                            elevators const& e,
                            osr::location const& pos,
                            osr::direction const dir,
                            std::vector<api::ModeEnum> const& modes,
                            api::PedestrianProfileEnum const pedestrian_profile,
                            api::ElevationCostsEnum const elevation_costs,
                            double const max_matching_distance,
                            std::chrono::seconds const max) {
  if (!r.w_ || !r.l_ || !r.pl_ || !r.tt_ || !r.loc_tree_ || !r.matches_) {
    return {};
  }

  auto ret = hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>{};
  for (auto const m : modes) {
    if (m == api::ModeEnum::ODM) {
      continue;
    }

    auto const profile = to_profile(m, pedestrian_profile, elevation_costs);

    if (profile != osr::search_profile::kWheelchair) {
      continue;  // handled by get_offsets
    }

    utl::equal_ranges_linear(
        get_td_footpaths(*r.w_, *r.l_, *r.pl_, *r.tt_, *r.loc_tree_, e,
                         *r.matches_, n::location_idx_t::invalid(), pos, dir,
                         profile, max, max_matching_distance, *blocked),
        [](n::td_footpath const& a, n::td_footpath const& b) {
          return a.target_ == b.target_;
        },
        [&](auto&& from, auto&& to) {
          ret.emplace(from->target_,
                      utl::to_vec(from, to, [&](n::td_footpath const fp) {
                        return n::routing::td_offset{
                            .valid_from_ = fp.valid_from_,
                            .duration_ = fp.duration_,
                            .transport_mode_id_ =
                                static_cast<n::transport_mode_id_t>(profile)};
                      }));
        });
  }

  return ret;
}

td_offsets_t routing::get_td_offsets(
    elevators const* e,
    place_t const& p,
    osr::direction const dir,
    std::vector<api::ModeEnum> const& modes,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs,
    double const max_matching_distance,
    std::chrono::seconds const max) const {
  return e != nullptr
             ? std::visit(
                   utl::overloaded{[&](tt_location) { return td_offsets_t{}; },
                                   [&](osr::location const& pos) {
                                     return ::motis::ep::get_td_offsets(
                                         *this, *e, pos, dir, modes,
                                         pedestrian_profile, elevation_costs,
                                         max_matching_distance, max);
                                   }},
                   p)
             : td_offsets_t{};
}

std::vector<n::routing::offset> get_offsets(
    routing const& r,
    osr::location const& pos,
    osr::direction const dir,
    osr::elevation_storage const* elevations,
    std::vector<api::ModeEnum> const& modes,
    std::optional<std::vector<api::RentalFormFactorEnum>> const& form_factors,
    std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&
        propulsion_types,
    std::optional<std::vector<std::string>> const& rental_providers,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs,
    std::chrono::seconds const max,
    double const max_matching_distance,
    gbfs::gbfs_routing_data& gbfs_rd) {
  if (!r.w_ || !r.l_ || !r.pl_ || !r.tt_ || !r.loc_tree_ || !r.matches_) {
    return {};
  }
  auto offsets = std::vector<n::routing::offset>{};
  auto ignore_walk = false;

  auto const handle_mode = [&](api::ModeEnum const m) {
    auto const profile = to_profile(m, pedestrian_profile, elevation_costs);

    if (r.rt_->e_ && profile == osr::search_profile::kWheelchair) {
      return;  // handled by get_td_offsets
    }

    auto const max_dist = get_max_distance(profile, max);

    auto const near_stops = r.loc_tree_->in_radius(pos.pos_, max_dist);
    auto const near_stop_locations =
        utl::to_vec(near_stops, [&](n::location_idx_t const l) {
          return osr::location{r.tt_->locations_.coordinates_[l],
                               r.pl_->get_level(*r.w_, (*r.matches_)[l])};
        });

    if (osr::is_rental_profile(profile)) {
      if (!gbfs_rd.has_data()) {
        return;
      }

      auto providers = hash_set<gbfs_provider_idx_t>{};
      gbfs_rd.data_->provider_rtree_.in_radius(
          pos.pos_, max_dist, [&](auto const pi) { providers.insert(pi); });

      for (auto const& pi : providers) {
        auto const& provider = gbfs_rd.data_->providers_.at(pi);
        if (provider == nullptr ||
            (rental_providers && utl::find(*rental_providers, provider->id_) ==
                                     end(*rental_providers))) {
          continue;
        }
        auto provider_rd = std::shared_ptr<gbfs::provider_routing_data>{};
        for (auto const& prod : provider->products_) {
          if (prod.return_constraint_ ==
                  gbfs::return_constraint::kRoundtripStation ||
              !gbfs::products_match(prod, form_factors, propulsion_types)) {
            continue;
          }
          if (!provider_rd) {
            provider_rd = gbfs_rd.get_provider_routing_data(*provider);
          }
          auto const prod_ref = gbfs::gbfs_products_ref{pi, prod.idx_};
          auto* prod_rd =
              gbfs_rd.get_products_routing_data(*provider, prod.idx_);
          auto const sharing = prod_rd->get_sharing_data(r.w_->n_nodes());
          auto const paths = osr::route(
              *r.w_, *r.l_, gbfs::get_osr_profile(prod), pos,
              near_stop_locations, static_cast<osr::cost_t>(max.count()), dir,
              kMaxMatchingDistance, nullptr, &sharing, elevations);
          ignore_walk = true;
          for (auto const [p, l] : utl::zip(paths, near_stops)) {
            if (p.has_value()) {
              offsets.emplace_back(l, n::duration_t{p->cost_ / 60},
                                   gbfs_rd.get_transport_mode(prod_ref));
            }
          }
        }
      }

    } else {
      auto const paths =
          osr::route(*r.w_, *r.l_, profile, pos, near_stop_locations,
                     static_cast<osr::cost_t>(max.count()), dir,
                     max_matching_distance, nullptr, nullptr, elevations);
      for (auto const [p, l] : utl::zip(paths, near_stops)) {
        if (p.has_value()) {
          offsets.emplace_back(l, n::duration_t{p->cost_ / 60},
                               static_cast<n::transport_mode_id_t>(profile));
        }
      }
    }
  };

  if (utl::find(modes, api::ModeEnum::RENTAL) != end(modes)) {
    handle_mode(api::ModeEnum::RENTAL);
  }

  for (auto const m : modes) {
    if (m == api::ModeEnum::RENTAL) {
      continue;  // handled above
    }
    if (m == api::ModeEnum::WALK && ignore_walk) {
      continue;
    }
    handle_mode(m);
  }

  return offsets;
}

std::vector<n::routing::offset> routing::get_offsets(
    place_t const& p,
    osr::direction const dir,
    std::vector<api::ModeEnum> const& modes,
    std::optional<std::vector<api::RentalFormFactorEnum>> const& form_factors,
    std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&
        propulsion_types,
    std::optional<std::vector<std::string>> const& rental_providers,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs,
    std::chrono::seconds const max,
    double const max_matching_distance,
    gbfs::gbfs_routing_data& gbfs_rd) const {
  return std::visit(
      utl::overloaded{[&](tt_location const l) { return station_start(l.l_); },
                      [&](osr::location const& pos) {
                        return ::motis::ep::get_offsets(
                            *this, pos, dir, elevations_, modes, form_factors,
                            propulsion_types, rental_providers,
                            pedestrian_profile, elevation_costs, max,
                            max_matching_distance, gbfs_rd);
                      }},
      p);
}

std::pair<n::routing::query, std::optional<n::unixtime_t>> get_start_time(
    api::plan_params const& query) {
  if (query.pageCursor_.has_value()) {
    return {cursor_to_query(*query.pageCursor_), std::nullopt};
  } else {
    auto const t = std::chrono::time_point_cast<n::i32_minutes>(
        *query.time_.value_or(openapi::now()));
    auto const window = std::chrono::duration_cast<n::duration_t>(
        std::chrono::seconds{query.searchWindow_ * (query.arriveBy_ ? -1 : 1)});
    return {{.start_time_ = query.timetableView_
                                ? n::routing::start_time_t{n::interval{
                                      query.arriveBy_ ? t - window : t,
                                      query.arriveBy_ ? t : t + window}}
                                : n::routing::start_time_t{t},
             .extend_interval_earlier_ = query.arriveBy_,
             .extend_interval_later_ = !query.arriveBy_},
            t};
  }
}

std::pair<std::vector<api::Itinerary>, n::duration_t> routing::route_direct(
    elevators const* e,
    gbfs::gbfs_routing_data& gbfs_rd,
    api::Place const& from,
    api::Place const& to,
    std::vector<api::ModeEnum> const& modes,
    std::optional<std::vector<api::RentalFormFactorEnum>> const& form_factors,
    std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&
        propulsion_types,
    std::optional<std::vector<std::string>> const& rental_providers,
    n::unixtime_t const start_time,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs,
    std::chrono::seconds max,
    double const max_matching_distance,
    double const fastest_direct_factor,
    unsigned const api_version) const {
  if (!w_ || !l_) {
    return {};
  }
  auto const omit_walk = gbfs_rd.has_data() &&
                         utl::find(modes, api::ModeEnum::RENTAL) != end(modes);
  auto fastest_direct = kInfinityDuration;
  auto cache = street_routing_cache_t{};
  auto itineraries = std::vector<api::Itinerary>{};
  for (auto const& m : modes) {
    if (m == api::ModeEnum::CAR || m == api::ModeEnum::BIKE ||
        m == api::ModeEnum::CAR_PARKING ||
        (!omit_walk && m == api::ModeEnum::WALK)) {
      auto itinerary = route(*w_, *l_, gbfs_rd, e, elevations_, from, to, m,
                             to_profile(m, pedestrian_profile, elevation_costs),
                             start_time, std::nullopt, max_matching_distance,
                             {}, cache, *blocked, api_version, max);
      if (itinerary.legs_.empty()) {
        continue;
      }
      auto const duration = std::chrono::duration_cast<n::duration_t>(
          std::chrono::seconds{itinerary.duration_});
      if (duration < fastest_direct) {
        fastest_direct = duration;
      }
      itineraries.emplace_back(std::move(itinerary));
    } else if (m == api::ModeEnum::RENTAL && gbfs_rd.has_data()) {
      // could be bike sharing or car sharing - car sharing has the higher max
      // distance, so we use this here to be safe
      auto const max_dist =
          get_max_distance(osr::search_profile::kCarSharing, max);
      auto providers = hash_set<gbfs_provider_idx_t>{};
      gbfs_rd.data_->provider_rtree_.in_radius(
          {from.lat_, from.lon_}, max_dist,
          [&](auto const pi) { providers.insert(pi); });
      for (auto const& pi : providers) {
        auto const& provider = gbfs_rd.data_->providers_.at(pi);
        if (provider == nullptr ||
            (rental_providers && utl::find(*rental_providers, provider->id_) ==
                                     end(*rental_providers))) {
          continue;
        }
        for (auto const& prod : provider->products_) {
          if (!gbfs::products_match(prod, form_factors, propulsion_types)) {
            continue;
          }
          auto const profile = gbfs::get_osr_profile(prod);
          auto itinerary =
              route(*w_, *l_, gbfs_rd, e, elevations_, from, to, m, profile,
                    start_time, std::nullopt, max_matching_distance,
                    gbfs::gbfs_products_ref{provider->idx_, prod.idx_}, cache,
                    *blocked, api_version, max);
          if (itinerary.legs_.empty()) {
            continue;
          }
          auto const duration = std::chrono::duration_cast<n::duration_t>(
              std::chrono::seconds{itinerary.duration_});
          if (duration < fastest_direct) {
            fastest_direct = duration;
          }
          itineraries.emplace_back(std::move(itinerary));
        }
      }
    }
  }
  return {itineraries, fastest_direct != kInfinityDuration
                           ? std::chrono::round<n::duration_t>(
                                 fastest_direct * fastest_direct_factor)
                           : fastest_direct};
}

using stats_map_t = std::map<std::string, std::uint64_t>;

stats_map_t join(auto&&... maps) {
  auto ret = std::map<std::string, std::uint64_t>{};
  auto const add = [&](std::map<std::string, std::uint64_t> const& x) {
    ret.insert(begin(x), end(x));
  };
  (add(maps), ...);
  return ret;
}

void remove_slower_than_fastest_direct(n::routing::query& q) {
  if (!q.fastest_direct_) {
    return;
  }

  constexpr auto const kMaxDuration =
      n::duration_t{std::numeric_limits<n::duration_t::rep>::max()};

  auto const worse_than_fastest_direct = [&](n::duration_t const min) {
    return [&, min](auto const& o) {
      return o.duration() + min >= q.fastest_direct_;
    };
  };
  auto const get_min_duration = [&](auto&& x) {
    return x.empty() ? kMaxDuration
                     : utl::min_element(x, [](auto&& a, auto&& b) {
                         return a.duration() < b.duration();
                       })->duration();
  };

  auto min_start = get_min_duration(q.start_);
  for (auto const& [_, v] : q.td_start_) {
    min_start = std::min(min_start, get_min_duration(v));
  }

  auto min_dest = get_min_duration(q.destination_);
  for (auto const& [_, v] : q.td_dest_) {
    min_dest = std::min(min_dest, get_min_duration(v));
  }

  utl::erase_if(q.start_, worse_than_fastest_direct(min_dest));
  utl::erase_if(q.destination_, worse_than_fastest_direct(min_start));
  for (auto& [k, v] : q.td_start_) {
    utl::erase_if(v, worse_than_fastest_direct(min_dest));
  }
  for (auto& [k, v] : q.td_dest_) {
    utl::erase_if(v, worse_than_fastest_direct(min_start));
  }
}

std::vector<n::routing::via_stop> get_via_stops(
    n::timetable const& tt,
    tag_lookup const& tags,
    std::optional<std::vector<std::string>> const& vias,
    std::vector<std::int64_t> const& times) {
  if (!vias.has_value()) {
    return {};
  }

  auto ret = std::vector<n::routing::via_stop>{};
  for (auto i = 0U; i != vias->size(); ++i) {
    ret.push_back({tags.get_location(tt, (*vias)[i]),
                   n::duration_t{i < times.size() ? times[i] : 0}});
  }
  return ret;
}

std::vector<api::ModeEnum> deduplicate(std::vector<api::ModeEnum> m) {
  utl::erase_duplicates(m);
  return m;
};

api::plan_response routing::operator()(boost::urls::url_view const& url) const {
  auto const rt = rt_;
  auto const rtt = rt->rtt_.get();
  auto const e = rt_->e_.get();
  auto gbfs_rd = gbfs::gbfs_routing_data{w_, l_, gbfs_};
  if (blocked.get() == nullptr && w_ != nullptr) {
    blocked.reset(new osr::bitvec<osr::node_idx_t>{w_->n_nodes()});
  }

  auto const query = api::plan_params{url.params()};
  auto const api_version = url.encoded_path().contains("/v1/") ? 1U : 2U;

  auto const deduplicate = [](auto m) {
    utl::erase_duplicates(m);
    return m;
  };
  auto const pre_transit_modes = deduplicate(query.preTransitModes_);
  auto const post_transit_modes = deduplicate(query.postTransitModes_);
  auto const direct_modes = deduplicate(query.directModes_);
  auto const from = get_place(tt_, tags_, query.fromPlace_);
  auto const to = get_place(tt_, tags_, query.toPlace_);
  auto const from_p = to_place(tt_, tags_, w_, pl_, matches_, from);
  auto const to_p = to_place(tt_, tags_, w_, pl_, matches_, to);

  auto const& start = query.arriveBy_ ? to : from;
  auto const& dest = query.arriveBy_ ? from : to;
  auto const& start_modes =
      query.arriveBy_ ? post_transit_modes : pre_transit_modes;
  auto const& dest_modes =
      query.arriveBy_ ? pre_transit_modes : post_transit_modes;
  auto const& start_form_factors = query.arriveBy_
                                       ? query.postTransitRentalFormFactors_
                                       : query.preTransitRentalFormFactors_;
  auto const& dest_form_factors = query.arriveBy_
                                      ? query.preTransitRentalFormFactors_
                                      : query.postTransitRentalFormFactors_;
  auto const& start_propulsion_types =
      query.arriveBy_ ? query.postTransitRentalPropulsionTypes_
                      : query.preTransitRentalPropulsionTypes_;
  auto const& dest_propulsion_types =
      query.arriveBy_ ? query.postTransitRentalPropulsionTypes_
                      : query.preTransitRentalPropulsionTypes_;
  auto const& start_rental_providers = query.arriveBy_
                                           ? query.postTransitRentalProviders_
                                           : query.preTransitRentalProviders_;
  auto const& dest_rental_providers = query.arriveBy_
                                          ? query.preTransitRentalProviders_
                                          : query.postTransitRentalProviders_;

  auto const [start_time, t] = get_start_time(query);

  UTL_START_TIMING(direct);
  auto [direct, fastest_direct] =
      t.has_value() && !direct_modes.empty() && w_ && l_
          ? route_direct(e, gbfs_rd, from_p, to_p, direct_modes,
                         query.directRentalFormFactors_,
                         query.directRentalPropulsionTypes_,
                         query.directRentalProviders_, *t,
                         query.pedestrianProfile_, query.elevationCosts_,
                         std::chrono::seconds{query.maxDirectTime_},
                         query.maxMatchingDistance_, query.fastestDirectFactor_,
                         api_version)
          : std::pair{std::vector<api::Itinerary>{}, kInfinityDuration};
  UTL_STOP_TIMING(direct);

  if (!query.transitModes_.empty() && fastest_direct > 5min) {
    utl::verify(tt_ != nullptr && tags_ != nullptr,
                "mode=TRANSIT requires timetable to be loaded");

    auto const with_odm_pre_transit =
        utl::find(pre_transit_modes, api::ModeEnum::ODM) !=
        end(pre_transit_modes);
    auto const with_odm_post_transit =
        utl::find(post_transit_modes, api::ModeEnum::ODM) !=
        end(post_transit_modes);
    auto const with_odm_direct =
        utl::find(direct_modes, api::ModeEnum::ODM) != end(direct_modes);

    if (with_odm_pre_transit || with_odm_post_transit || with_odm_direct) {
      utl::verify(config_.has_odm(), "ODM not configured");
      return odm::meta_router{*this,
                              query,
                              pre_transit_modes,
                              post_transit_modes,
                              direct_modes,
                              from,
                              to,
                              from_p,
                              to_p,
                              start_time,
                              direct,
                              fastest_direct,
                              with_odm_pre_transit,
                              with_odm_post_transit,
                              with_odm_direct,
                              api_version}
          .run();
    }

    UTL_START_TIMING(query_preparation);
    auto q = n::routing::query{
        .start_time_ = start_time.start_time_,
        .start_match_mode_ = get_match_mode(start),
        .dest_match_mode_ = get_match_mode(dest),
        .use_start_footpaths_ = !is_intermodal(start),
        .start_ = get_offsets(start,
                              query.arriveBy_ ? osr::direction::kBackward
                                              : osr::direction::kForward,
                              start_modes, start_form_factors,
                              start_propulsion_types, start_rental_providers,
                              query.pedestrianProfile_, query.elevationCosts_,
                              std::chrono::seconds{query.maxPreTransitTime_},
                              query.maxMatchingDistance_, gbfs_rd),
        .destination_ =
            get_offsets(dest,
                        query.arriveBy_ ? osr::direction::kForward
                                        : osr::direction::kBackward,
                        dest_modes, dest_form_factors, dest_propulsion_types,
                        dest_rental_providers, query.pedestrianProfile_,
                        query.elevationCosts_,
                        std::chrono::seconds{query.maxPostTransitTime_},
                        query.maxMatchingDistance_, gbfs_rd),
        .td_start_ =
            get_td_offsets(e, start,
                           query.arriveBy_ ? osr::direction::kBackward
                                           : osr::direction::kForward,
                           start_modes, query.pedestrianProfile_,
                           query.elevationCosts_, query.maxMatchingDistance_,
                           std::chrono::seconds{query.maxPreTransitTime_}),
        .td_dest_ =
            get_td_offsets(e, dest,
                           query.arriveBy_ ? osr::direction::kForward
                                           : osr::direction::kBackward,
                           dest_modes, query.pedestrianProfile_,
                           query.elevationCosts_, query.maxMatchingDistance_,
                           std::chrono::seconds{query.maxPostTransitTime_}),
        .max_transfers_ = static_cast<std::uint8_t>(
            query.maxTransfers_.has_value() ? *query.maxTransfers_
                                            : n::routing::kMaxTransfers),
        .max_travel_time_ = query.maxTravelTime_
                                .and_then([](std::int64_t const dur) {
                                  return std::optional{n::duration_t{dur}};
                                })
                                .value_or(kInfinityDuration),
        .min_connection_count_ = static_cast<unsigned>(query.numItineraries_),
        .extend_interval_earlier_ = start_time.extend_interval_earlier_,
        .extend_interval_later_ = start_time.extend_interval_later_,
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
                .additional_time_ =
                    n::duration_t{query.additionalTransferTime_},
                .factor_ = static_cast<float>(query.transferTimeFactor_)},
        .via_stops_ =
            get_via_stops(*tt_, *tags_, query.via_, query.viaMinimumStay_),
        .fastest_direct_ = fastest_direct == kInfinityDuration
                               ? std::nullopt
                               : std::optional{fastest_direct}};
    remove_slower_than_fastest_direct(q);
    UTL_STOP_TIMING(query_preparation);

    if (tt_->locations_.footpaths_out_.at(q.prf_idx_).empty()) {
      q.prf_idx_ = 0U;
    }

    if (search_state.get() == nullptr) {
      search_state.reset(new n::routing::search_state{});
    }
    if (raptor_state.get() == nullptr) {
      raptor_state.reset(new n::routing::raptor_state{});
    }

    auto const query_stats =
        stats_map_t{{"direct", UTL_TIMING_MS(direct)},
                    {"query_preparation", UTL_TIMING_MS(query_preparation)},
                    {"n_start_offsets", q.start_.size()},
                    {"n_dest_offsets", q.destination_.size()},
                    {"n_td_start_offsets", q.td_start_.size()},
                    {"n_td_dest_offsets", q.td_dest_.size()}};

    auto const r = n::routing::raptor_search(
        *tt_, rtt, *search_state, *raptor_state, std::move(q),
        query.arriveBy_ ? n::direction::kBackward : n::direction::kForward,
        query.timeout_.has_value()
            ? std::optional<std::chrono::seconds>{*query.timeout_}
            : std::nullopt);

    return {
        .debugOutput_ = join(std::move(query_stats), r.search_stats_.to_map(),
                             r.algo_stats_.to_map()),
        .from_ = from_p,
        .to_ = to_p,
        .direct_ = std::move(direct),
        .itineraries_ = utl::to_vec(
            *r.journeys_,
            [&, cache = street_routing_cache_t{}](auto&& j) mutable {
              return journey_to_response(
                  w_, l_, pl_, *tt_, *tags_, e, rtt, matches_, elevations_,
                  shapes_, gbfs_rd, query.pedestrianProfile_,
                  query.elevationCosts_, j, start, dest, cache, blocked.get(),
                  query.detailedTransfers_, query.withFares_,
                  config_.timetable_
                      .and_then([](config::timetable const& x) {
                        return std::optional{x.max_matching_distance_};
                      })
                      .value_or(kMaxMatchingDistance),
                  query.maxMatchingDistance_, api_version);
            }),
        .previousPageCursor_ =
            fmt::format("EARLIER|{}", to_seconds(r.interval_.from_)),
        .nextPageCursor_ = fmt::format("LATER|{}", to_seconds(r.interval_.to_)),
    };
  }

  return {.from_ = to_place(tt_, tags_, w_, pl_, matches_, from),
          .to_ = to_place(tt_, tags_, w_, pl_, matches_, to),
          .direct_ = std::move(direct),
          .itineraries_ = {}};
}

}  // namespace motis::ep
