#include "motis/endpoints/routing.h"

#include <cmath>
#include <algorithm>

#include "boost/thread/tss.hpp"

#include "openapi/bad_request_exception.h"

#include "prometheus/counter.h"
#include "prometheus/histogram.h"

#include "utl/erase_duplicates.h"
#include "utl/helpers/algorithm.h"
#include "utl/timing.h"

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/routing/profile.h"
#include "osr/routing/profiles/car.h"
#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"
#include "osr/routing/sharing_data.h"
#include "osr/types.h"

#include "nigiri/common/interval.h"
#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/special_stations.h"

#include "motis/constants.h"
#include "motis/endpoints/routing.h"

#include "nigiri/routing/raptor/pong.h"
#include "nigiri/routing/tb/query_engine.h"
#include "nigiri/routing/tb/tb_data.h"
#include "nigiri/routing/tb/tb_search.h"

#include "motis/config.h"
#include "motis/direct_filter.h"
#include "motis/flex/flex.h"
#include "motis/flex/flex_output.h"
#include "motis/gbfs/data.h"
#include "motis/gbfs/gbfs_output.h"
#include "motis/gbfs/mode.h"
#include "motis/gbfs/osr_profile.h"
#include "motis/get_stops_with_traffic.h"
#include "motis/journey_to_response.h"
#include "motis/match_platforms.h"
#include "motis/metrics_registry.h"
#include "motis/odm/meta_router.h"
#include "motis/osr/max_distance.h"
#include "motis/osr/mode_to_profile.h"
#include "motis/osr/street_routing.h"
#include "motis/parse_location.h"
#include "motis/server.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/modes_to_clasz_mask.h"
#include "motis/timetable/time_conv.h"
#include "motis/update_rtt_td_footpaths.h"

namespace n = nigiri;
using namespace std::chrono_literals;

namespace motis::ep {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<osr::bitvec<osr::node_idx_t>> blocked;

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

n::routing::td_offsets_t get_td_offsets(
    routing const& r,
    n::rt_timetable const* rtt,
    elevators const* e,
    osr::location const& pos,
    osr::direction const dir,
    std::vector<api::ModeEnum> const& modes,
    osr_parameters const& osr_params,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs,
    double const max_matching_distance,
    std::chrono::seconds const max,
    nigiri::routing::start_time_t const& start_time,
    stats_map_t& stats) {
  if (!r.w_ || !r.l_ || !r.pl_ || !r.tt_ || !r.loc_tree_ || !r.matches_) {
    return {};
  }

  auto ret = hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>{};
  for (auto const m : modes) {
    if (m == api::ModeEnum::ODM || m == api::ModeEnum::RIDE_SHARING) {
      continue;
    } else if (m == api::ModeEnum::FLEX) {
      UTL_START_TIMING(flex_timer);
      utl::verify(r.fa_, "FLEX areas not loaded");
      auto frd = flex::flex_routing_data{};
      flex::add_flex_td_offsets(*r.w_, *r.l_, r.pl_, r.matches_, r.way_matches_,
                                *r.tt_, *r.fa_, *r.loc_tree_, start_time, pos,
                                dir, max, max_matching_distance, osr_params,
                                frd, ret, stats);
      stats.emplace(fmt::format("prepare_{}_FLEX", to_str(dir)),
                    UTL_GET_TIMING_MS(flex_timer));
      continue;
    }

    auto const profile = to_profile(m, pedestrian_profile, elevation_costs);

    if (e == nullptr || profile != osr::search_profile::kWheelchair) {
      continue;  // handled by get_offsets
    }

    utl::equal_ranges_linear(
        get_td_footpaths(*r.w_, *r.l_, *r.pl_, *r.tt_, rtt, *r.loc_tree_, *e,
                         *r.matches_, n::location_idx_t::invalid(), pos, dir,
                         profile, max, max_matching_distance, osr_params,
                         *blocked),
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

n::routing::td_offsets_t routing::get_td_offsets(
    n::rt_timetable const* rtt,
    elevators const* e,
    place_t const& p,
    osr::direction const dir,
    std::vector<api::ModeEnum> const& modes,
    osr_parameters const& osr_params,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs,
    double const max_matching_distance,
    std::chrono::seconds const max,
    nigiri::routing::start_time_t const& start_time,
    stats_map_t& stats) const {
  return std::visit(
      utl::overloaded{[&](tt_location) { return n::routing::td_offsets_t{}; },
                      [&](osr::location const& pos) {
                        return ::motis::ep::get_td_offsets(
                            *this, rtt, e, pos, dir, modes, osr_params,
                            pedestrian_profile, elevation_costs,
                            max_matching_distance, max, start_time, stats);
                      }},
      p);
}

bool include_rental_provider(
    std::optional<std::vector<std::string>> const& rental_providers,
    std::optional<std::vector<std::string>> const& rental_provider_groups,
    gbfs::gbfs_provider const* provider) {
  if (provider == nullptr) {
    return false;
  }
  if ((!rental_providers || rental_providers->empty()) &&
      (!rental_provider_groups || rental_provider_groups->empty())) {
    return true;
  }
  return (rental_provider_groups &&
          utl::find(*rental_provider_groups, provider->group_id_) !=
              end(*rental_provider_groups)) ||
         (rental_providers && utl::find(*rental_providers, provider->id_) !=
                                  end(*rental_providers));
}

std::vector<n::routing::offset> get_offsets(
    routing const& r,
    n::rt_timetable const* rtt,
    osr::location const& pos,
    osr::direction const dir,
    osr::elevation_storage const* elevations,
    std::vector<api::ModeEnum> const& modes,
    std::optional<std::vector<api::RentalFormFactorEnum>> const& form_factors,
    std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&
        propulsion_types,
    std::optional<std::vector<std::string>> const& rental_providers,
    std::optional<std::vector<std::string>> const& rental_provider_groups,
    bool const ignore_rental_return_constraints,
    osr_parameters const& osr_params,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs,
    std::chrono::seconds const max,
    double const max_matching_distance,
    gbfs::gbfs_routing_data& gbfs_rd,
    stats_map_t& stats) {
  if (!r.w_ || !r.l_ || !r.pl_ || !r.tt_ || !r.loc_tree_ || !r.matches_) {
    return {};
  }
  auto offsets = std::vector<n::routing::offset>{};
  auto ignore_walk = false;

  auto const handle_mode = [&](api::ModeEnum const m) {
    UTL_START_TIMING(timer);

    auto profile = to_profile(m, pedestrian_profile, elevation_costs);

    if (r.rt_->e_ && profile == osr::search_profile::kWheelchair) {
      return;  // handled by get_td_offsets
    }

    if (osr::is_rental_profile(profile) &&
        (!form_factors.has_value() ||
         utl::any_of(*form_factors, [](auto const f) {
           return gbfs::get_osr_profile(gbfs::from_api_form_factor(f)) ==
                  osr::search_profile::kCarSharing;
         }))) {
      profile = osr::search_profile::kCarSharing;
    }

    auto const max_dist = get_max_distance(profile, max);
    auto const near_stops =
        get_stops_with_traffic(*r.tt_, rtt, *r.loc_tree_, pos, max_dist);
    auto const near_stop_locations =
        utl::to_vec(near_stops, [&](n::location_idx_t const l) {
          return osr::location{r.tt_->locations_.coordinates_[l],
                               r.pl_->get_level(*r.w_, (*r.matches_)[l])};
        });

    auto const route = [&](osr::search_profile const p,
                           osr::sharing_data const* sharing) {
      auto const params = to_profile_parameters(p, osr_params);
      auto const pos_match = r.l_->match(params, pos, false, dir,
                                         max_matching_distance, nullptr, p);
      auto const near_stop_matches = get_reverse_platform_way_matches(
          *r.l_, r.way_matches_, p, near_stops, near_stop_locations, dir,
          max_matching_distance);
      return osr::route(params, *r.w_, *r.l_, p, pos, near_stop_locations,
                        pos_match, near_stop_matches,
                        static_cast<osr::cost_t>(max.count()), dir, nullptr,
                        sharing, elevations);
    };

    if (osr::is_rental_profile(profile)) {
      if (!gbfs_rd.has_data()) {
        return;
      }

      auto const max_dist_to_departure =
          dir == osr::direction::kForward
              ? get_max_distance(osr::search_profile::kFoot, max)
              : max_dist;
      auto providers = hash_set<gbfs_provider_idx_t>{};
      gbfs_rd.data_->provider_rtree_.in_radius(
          pos.pos_, max_dist_to_departure,
          [&](auto const pi) { providers.insert(pi); });

      for (auto const& pi : providers) {
        UTL_START_TIMING(provider_timer);

        auto const& provider = gbfs_rd.data_->providers_.at(pi);
        if (!include_rental_provider(rental_providers, rental_provider_groups,
                                     provider.get())) {
          continue;
        }
        auto provider_rd = std::shared_ptr<gbfs::provider_routing_data>{};
        for (auto const& prod : provider->products_) {
          if ((prod.return_constraint_ ==
                   gbfs::return_constraint::kRoundtripStation &&
               !ignore_rental_return_constraints) ||
              !gbfs::products_match(prod, form_factors, propulsion_types)) {
            continue;
          }
          if (!provider_rd) {
            provider_rd = gbfs_rd.get_provider_routing_data(*provider);
          }
          auto const prod_ref = gbfs::gbfs_products_ref{pi, prod.idx_};
          auto* prod_rd =
              gbfs_rd.get_products_routing_data(*provider, prod.idx_);
          auto const sharing = prod_rd->get_sharing_data(
              r.w_->n_nodes(), ignore_rental_return_constraints);

          auto const paths =
              route(gbfs::get_osr_profile(prod.form_factor_), &sharing);
          ignore_walk = true;
          for (auto const [p, l] : utl::zip(paths, near_stops)) {
            if (p.has_value()) {
              offsets.emplace_back(l,
                                   n::duration_t{static_cast<unsigned>(
                                       std::ceil(p->cost_ / 60.0))},
                                   gbfs_rd.get_transport_mode(prod_ref));
            }
          }
        }

        stats.emplace(fmt::format("prepare_{}_{}_{}", to_str(dir),
                                  fmt::streamed(m), provider->id_),
                      UTL_GET_TIMING_MS(provider_timer));
      }

    } else {
      auto const paths = route(profile, nullptr);
      for (auto const [p, l] : utl::zip(paths, near_stops)) {
        if (p.has_value()) {
          offsets.emplace_back(
              l,
              n::duration_t{static_cast<unsigned>(std::ceil(p->cost_ / 60.0))},
              static_cast<n::transport_mode_id_t>(profile));
        }
      }
    }

    stats.emplace(fmt::format("prepare_{}_{}", to_str(dir), fmt::streamed(m)),
                  UTL_GET_TIMING_MS(timer));
  };

  if (utl::find(modes, api::ModeEnum::RENTAL) != end(modes)) {
    handle_mode(api::ModeEnum::RENTAL);
  }

  for (auto const m : modes) {
    if (m == api::ModeEnum::RENTAL || m == api::ModeEnum::FLEX ||
        (m == api::ModeEnum::WALK && ignore_walk)) {
      continue;
    }
    handle_mode(m);
  }

  return offsets;
}

n::interval<n::unixtime_t> shrink(bool const keep_late,
                                  std::size_t const max_size,
                                  n::interval<n::unixtime_t> search_interval,
                                  std::vector<n::routing::journey>& journeys) {
  if (journeys.size() <= max_size) {
    return search_interval;
  }

  if (keep_late) {
    auto cutoff_it =
        std::next(journeys.rbegin(), static_cast<int>(max_size - 1));
    auto last_arr_time = cutoff_it->start_time_;
    ++cutoff_it;
    while (cutoff_it != rend(journeys) &&
           cutoff_it->start_time_ == last_arr_time) {
      ++cutoff_it;
    }
    if (cutoff_it == rend(journeys)) {
      return search_interval;
    }
    search_interval.from_ = cutoff_it->start_time_ + std::chrono::minutes{1};
    journeys.erase(begin(journeys), cutoff_it.base());
  } else {
    auto cutoff_it = std::next(begin(journeys), static_cast<int>(max_size - 1));
    auto last_dep_time = cutoff_it->start_time_;
    while (cutoff_it != end(journeys) &&
           cutoff_it->start_time_ == last_dep_time) {
      ++cutoff_it;
    }
    if (cutoff_it == end(journeys)) {
      return search_interval;
    }
    search_interval.to_ = cutoff_it->start_time_;
    journeys.erase(cutoff_it, end(journeys));
  }

  return search_interval;
}

std::vector<n::routing::offset> routing::get_offsets(
    n::rt_timetable const* rtt,
    place_t const& p,
    osr::direction const dir,
    std::vector<api::ModeEnum> const& modes,
    std::optional<std::vector<api::RentalFormFactorEnum>> const& form_factors,
    std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&
        propulsion_types,
    std::optional<std::vector<std::string>> const& rental_providers,
    std::optional<std::vector<std::string>> const& rental_provider_groups,
    bool const ignore_rental_return_constraints,
    osr_parameters const& osr_params,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs,
    std::chrono::seconds const max,
    double const max_matching_distance,
    gbfs::gbfs_routing_data& gbfs_rd,
    stats_map_t& stats) const {
  return std::visit(
      utl::overloaded{[&](tt_location const l) { return station_start(l.l_); },
                      [&](osr::location const& pos) {
                        return ::motis::ep::get_offsets(
                            *this, rtt, pos, dir, elevations_, modes,
                            form_factors, propulsion_types, rental_providers,
                            rental_provider_groups,
                            ignore_rental_return_constraints, osr_params,
                            pedestrian_profile, elevation_costs, max,
                            max_matching_distance, gbfs_rd, stats);
                      }},
      p);
}

std::pair<n::routing::query, std::optional<n::unixtime_t>> get_start_time(
    api::plan_params const& query, nigiri::timetable const* tt) {
  if (query.pageCursor_.has_value()) {
    return {cursor_to_query(*query.pageCursor_), std::nullopt};
  } else {
    auto const t = std::chrono::time_point_cast<n::i32_minutes>(
        *query.time_.value_or(openapi::now()));
    utl::verify<openapi::bad_request_exception>(
        tt == nullptr || tt->external_interval().contains(t),
        "query time {} is outside of loaded timetable window {}", t,
        tt ? tt->external_interval() : n::interval<n::unixtime_t>{});
    auto const window =
        std::chrono::duration_cast<n::duration_t>(std::chrono::seconds{
            query.searchWindow_ *
            (query.arriveBy_ ? -1 : 1)});  // TODO redundant minus
    return {{.start_time_ = query.timetableView_ && tt
                                ? n::routing::start_time_t{n::interval{
                                      tt->external_interval().clamp(
                                          query.arriveBy_ ? t - window : t),
                                      tt->external_interval().clamp(
                                          query.arriveBy_ ? t + n::duration_t{1}
                                                          : t + window)}}
                                : n::routing::start_time_t{t},
             .extend_interval_earlier_ = query.arriveBy_,
             .extend_interval_later_ = !query.arriveBy_},
            t};
  }
}

std::pair<std::vector<api::Itinerary>, n::duration_t> routing::route_direct(
    elevators const* e,
    gbfs::gbfs_routing_data& gbfs_rd,
    n::lang_t const& lang,
    api::Place const& from,
    api::Place const& to,
    std::vector<api::ModeEnum> const& modes,
    std::optional<std::vector<api::RentalFormFactorEnum>> const& form_factors,
    std::optional<std::vector<api::RentalPropulsionTypeEnum>> const&
        propulsion_types,
    std::optional<std::vector<std::string>> const& rental_providers,
    std::optional<std::vector<std::string>> const& rental_provider_groups,
    bool const ignore_rental_return_constraints,
    n::unixtime_t const time,
    bool const arrive_by,
    osr_parameters const& osr_params,
    api::PedestrianProfileEnum const pedestrian_profile,
    api::ElevationCostsEnum const elevation_costs,
    std::chrono::seconds max,
    double const max_matching_distance,
    double const fastest_direct_factor,
    unsigned const api_version) const {
  if (!w_ || !l_) {
    return {};
  }
  auto fastest_direct = kInfinityDuration;
  auto cache = street_routing_cache_t{};
  auto itineraries = std::vector<api::Itinerary>{};

  auto const route_with_profile = [&](output const& out) {
    auto itinerary = street_routing(
        *w_, *l_, e, elevations_, lang, from, to, out,
        arrive_by ? std::nullopt : std::optional{time},
        arrive_by ? std::optional{time} : std::nullopt, max_matching_distance,
        osr_params, cache, *blocked, api_version, max);
    if (itinerary.legs_.empty()) {
      return false;
    }
    auto const duration = std::chrono::duration_cast<n::duration_t>(
        std::chrono::seconds{itinerary.duration_});
    if (duration < fastest_direct) {
      fastest_direct = duration;
    }
    itineraries.emplace_back(std::move(itinerary));
    return true;
  };

  for (auto const& m : modes) {
    if (m == api::ModeEnum::FLEX) {
      utl::verify(tt_ && tags_ && fa_, "FLEX requires timetable");
      auto const routings = flex::get_flex_routings(
          *tt_, *loc_tree_, time, get_location(from).pos_,
          osr::direction::kForward, max);
      for (auto const& [_, ids] : routings) {
        route_with_profile(flex::flex_output{*w_, *l_, pl_, matches_, ae_, tz_,
                                             *tags_, *tt_, *fa_, ids.front()});
      }
    } else if (m == api::ModeEnum::CAR || m == api::ModeEnum::BIKE ||
               m == api::ModeEnum::CAR_PARKING ||
               m == api::ModeEnum::CAR_DROPOFF || m == api::ModeEnum::WALK) {
      route_with_profile(default_output{
          *w_, to_profile(m, pedestrian_profile, elevation_costs)});
    } else if (m == api::ModeEnum::RENTAL && gbfs_rd.has_data()) {
      // use foot because this is always forward search and we need to walk to
      // the station/vehicle
      auto const max_dist = get_max_distance(osr::search_profile::kFoot, max);
      auto providers = hash_set<gbfs_provider_idx_t>{};
      auto routed = 0U;
      gbfs_rd.data_->provider_rtree_.in_radius(
          {from.lat_, from.lon_}, max_dist,
          [&](auto const pi) { providers.insert(pi); });
      for (auto const& pi : providers) {
        auto const& provider = gbfs_rd.data_->providers_.at(pi);
        if (!include_rental_provider(rental_providers, rental_provider_groups,
                                     provider.get())) {
          continue;
        }
        for (auto const& prod : provider->products_) {
          if (!gbfs::products_match(prod, form_factors, propulsion_types)) {
            continue;
          }
          route_with_profile(gbfs::gbfs_output{
              *w_, gbfs_rd, gbfs::gbfs_products_ref{provider->idx_, prod.idx_},
              ignore_rental_return_constraints});
          ++routed;
        }
      }
      // if we omitted the WALK routing but didn't have any rental providers
      // in the area, we need to do WALK routing now
      if (routed == 0U && utl::find(modes, api::ModeEnum::WALK) != end(modes)) {
        route_with_profile(default_output{
            *w_, to_profile(api::ModeEnum::WALK, pedestrian_profile,
                            elevation_costs)});
      }
    }
  }
  utl::erase_duplicates(itineraries);
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
      return o.duration() < nigiri::footpath::kMaxDuration &&
             o.duration() + min >= q.fastest_direct_;
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
    std::vector<std::int64_t> const& times,
    bool const reverse) {
  if (!vias.has_value()) {
    return {};
  }

  auto ret = std::vector<n::routing::via_stop>{};
  for (auto i = 0U; i != vias->size(); ++i) {
    ret.push_back({tags.get_location(tt, (*vias)[i]),
                   n::duration_t{i < times.size() ? times[i] : 0}});
  }

  if (reverse) {
    std::reverse(begin(ret), end(ret));
  }
  return ret;
}

std::vector<api::ModeEnum> deduplicate(std::vector<api::ModeEnum> m) {
  utl::erase_duplicates(m);
  return m;
};

api::plan_response routing::operator()(boost::urls::url_view const& url) const {
  metrics_->routing_requests_.Increment();

  auto const query = api::plan_params{url.params()};
  utl::verify<openapi::bad_request_exception>(
      !query.maxItineraries_.has_value() ||
          (*query.maxItineraries_ >= 1 &&
           *query.maxItineraries_ >= query.numItineraries_),
      "maxItineraries={} < numItineraries={}",
      query.maxItineraries_.value_or(0), query.numItineraries_);

  auto const rt = rt_;
  auto const rtt = rt->rtt_.get();
  auto const e = rt_->e_.get();
  auto gbfs_rd = gbfs::gbfs_routing_data{w_, l_, gbfs_};
  if (blocked.get() == nullptr && w_ != nullptr) {
    blocked.reset(new osr::bitvec<osr::node_idx_t>{w_->n_nodes()});
  }

  auto const api_version = get_api_version(url);

  auto const deduplicate = [](auto m) {
    utl::erase_duplicates(m);
    return m;
  };
  auto const& lang = query.language_;
  auto const pre_transit_modes = deduplicate(query.preTransitModes_);
  auto const post_transit_modes = deduplicate(query.postTransitModes_);
  auto const direct_modes = deduplicate(query.directModes_);
  auto const from = get_place(tt_, tags_, query.fromPlace_);
  auto const to = get_place(tt_, tags_, query.toPlace_);
  auto from_p = to_place(tt_, tags_, w_, pl_, matches_, ae_, tz_, lang, from);
  auto to_p = to_place(tt_, tags_, w_, pl_, matches_, ae_, tz_, lang, to);
  if (from_p.vertexType_ == api::VertexTypeEnum::NORMAL) {
    from_p.name_ = "START";
  }
  if (to_p.vertexType_ == api::VertexTypeEnum::NORMAL) {
    to_p.name_ = "END";
  }

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
  auto const& start_rental_provider_groups =
      query.arriveBy_ ? query.postTransitRentalProviderGroups_
                      : query.preTransitRentalProviderGroups_;
  auto const& dest_rental_provider_groups =
      query.arriveBy_ ? query.preTransitRentalProviderGroups_
                      : query.postTransitRentalProviderGroups_;
  auto const start_ignore_return_constraints =
      query.arriveBy_ ? query.ignorePostTransitRentalReturnConstraints_
                      : query.ignorePreTransitRentalReturnConstraints_;
  auto const dest_ignore_return_constraints =
      query.arriveBy_ ? query.ignorePreTransitRentalReturnConstraints_
                      : query.ignorePostTransitRentalReturnConstraints_;

  utl::verify(query.searchWindow_ / 60 <
                  config_.limits_.value().plan_max_search_window_minutes_,
              "maximum searchWindow size exceeded");

  auto const max_transfers =
      query.maxTransfers_.has_value() &&
              *query.maxTransfers_ <= n::routing::kMaxTransfers
          ? (*query.maxTransfers_ - (api_version < 3 ? 1 : 0))
          : n::routing::kMaxTransfers;
  auto const osr_params = get_osr_parameters(query);

  auto const [start_time, t] = get_start_time(query, tt_);

  UTL_START_TIMING(direct);
  auto [direct, fastest_direct] =
      t.has_value() && !direct_modes.empty() && w_ && l_
          ? route_direct(
                e, gbfs_rd, lang, from_p, to_p, direct_modes,
                query.directRentalFormFactors_,
                query.directRentalPropulsionTypes_,
                query.directRentalProviders_, query.directRentalProviderGroups_,
                query.ignoreDirectRentalReturnConstraints_, *t, query.arriveBy_,
                osr_params, query.pedestrianProfile_, query.elevationCosts_,
                std::min(std::chrono::seconds{query.maxDirectTime_},
                         std::chrono::seconds{
                             config_.limits_.value()
                                 .street_routing_max_direct_seconds_}),
                query.maxMatchingDistance_, query.fastestDirectFactor_,
                api_version)
          : std::pair{std::vector<api::Itinerary>{}, kInfinityDuration};
  UTL_STOP_TIMING(direct);

  if (!query.transitModes_.empty() && fastest_direct > 5min &&
      max_transfers >= 0) {
    utl::verify(tt_ != nullptr && tags_ != nullptr,
                "mode=TRANSIT requires timetable to be loaded");

    auto const max_results = config_.limits_.value().plan_max_results_;
    utl::verify(query.numItineraries_ <= max_results,
                "maximum number of minimum itineraries is {}", max_results);
    auto const max_timeout = std::chrono::seconds{
        config_.limits_.value().routing_max_timeout_seconds_};
    utl::verify(!query.timeout_.has_value() ||
                    std::chrono::seconds{*query.timeout_} <= max_timeout,
                "maximum allowed timeout is {}", max_timeout);

    auto const with_odm_pre_transit =
        utl::find(pre_transit_modes, api::ModeEnum::ODM) !=
        end(pre_transit_modes);
    auto const with_odm_post_transit =
        utl::find(post_transit_modes, api::ModeEnum::ODM) !=
        end(post_transit_modes);
    auto const with_odm_direct =
        utl::find(direct_modes, api::ModeEnum::ODM) != end(direct_modes);
    auto const with_ride_sharing_pre_transit =
        utl::find(pre_transit_modes, api::ModeEnum::RIDE_SHARING) !=
        end(pre_transit_modes);
    auto const with_ride_sharing_post_transit =
        utl::find(post_transit_modes, api::ModeEnum::RIDE_SHARING) !=
        end(post_transit_modes);
    auto const with_ride_sharing_direct =
        utl::find(direct_modes, api::ModeEnum::RIDE_SHARING) !=
        end(direct_modes);

    if (with_odm_pre_transit || with_odm_post_transit || with_odm_direct ||
        with_ride_sharing_pre_transit || with_ride_sharing_post_transit ||
        with_ride_sharing_direct) {
      utl::verify(config_.has_prima(), "PRIMA not configured");
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
                              with_ride_sharing_pre_transit,
                              with_ride_sharing_post_transit,
                              with_ride_sharing_direct,
                              api_version}
          .run();
    }

    auto const pre_transit_time = std::min(
        std::chrono::seconds{query.maxPreTransitTime_},
        std::chrono::seconds{config_.limits_.value()
                                 .street_routing_max_prepost_transit_seconds_});
    auto const post_transit_time = std::min(
        std::chrono::seconds{query.maxPostTransitTime_},
        std::chrono::seconds{config_.limits_.value()
                                 .street_routing_max_prepost_transit_seconds_});

    UTL_START_TIMING(query_preparation);
    auto prepare_stats = std::map<std::string, std::uint64_t>{};
    auto q = n::routing::query{
        .start_time_ = start_time.start_time_,
        .start_match_mode_ = get_match_mode(start),
        .dest_match_mode_ = get_match_mode(dest),
        .use_start_footpaths_ = !is_intermodal(start),
        .start_ = get_offsets(
            rtt, start,
            query.arriveBy_ ? osr::direction::kBackward
                            : osr::direction::kForward,
            start_modes, start_form_factors, start_propulsion_types,
            start_rental_providers, start_rental_provider_groups,
            start_ignore_return_constraints, osr_params,
            query.pedestrianProfile_, query.elevationCosts_, pre_transit_time,
            query.maxMatchingDistance_, gbfs_rd, prepare_stats),
        .destination_ = get_offsets(
            rtt, dest,
            query.arriveBy_ ? osr::direction::kForward
                            : osr::direction::kBackward,
            dest_modes, dest_form_factors, dest_propulsion_types,
            dest_rental_providers, dest_rental_provider_groups,
            dest_ignore_return_constraints, osr_params,
            query.pedestrianProfile_, query.elevationCosts_, post_transit_time,
            query.maxMatchingDistance_, gbfs_rd, prepare_stats),
        .td_start_ = get_td_offsets(
            rtt, e, start,
            query.arriveBy_ ? osr::direction::kBackward
                            : osr::direction::kForward,
            start_modes, osr_params, query.pedestrianProfile_,
            query.elevationCosts_, query.maxMatchingDistance_, pre_transit_time,
            start_time.start_time_, prepare_stats),
        .td_dest_ = get_td_offsets(
            rtt, e, dest,
            query.arriveBy_ ? osr::direction::kForward
                            : osr::direction::kBackward,
            dest_modes, osr_params, query.pedestrianProfile_,
            query.elevationCosts_, query.maxMatchingDistance_,
            post_transit_time, start_time.start_time_, prepare_stats),
        .max_transfers_ = static_cast<std::uint8_t>(max_transfers),
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
                ? query.requireCarTransport_ ? n::kCarProfile
                  : query.pedestrianProfile_ ==
                          api::PedestrianProfileEnum::WHEELCHAIR
                      ? n::kWheelchairProfile
                      : n::kFootProfile
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
                .additional_time_ =
                    n::duration_t{query.additionalTransferTime_},
                .factor_ = static_cast<float>(query.transferTimeFactor_)},
        .via_stops_ = get_via_stops(*tt_, *tags_, query.via_,
                                    query.viaMinimumStay_, query.arriveBy_),
        .fastest_direct_ = fastest_direct == kInfinityDuration
                               ? std::nullopt
                               : std::optional{fastest_direct},
        .fastest_direct_factor_ = query.fastestDirectFactor_,
        .slow_direct_ = query.slowDirect_,
        .fastest_slow_direct_factor_ = query.fastestSlowDirectFactor_};
    remove_slower_than_fastest_direct(q);
    UTL_STOP_TIMING(query_preparation);

    if (tt_->locations_.footpaths_out_.at(q.prf_idx_).empty()) {
      q.prf_idx_ = 0U;
    }

    auto const query_stats =
        stats_map_t{{"direct", UTL_TIMING_MS(direct)},
                    {"prepare", UTL_TIMING_MS(query_preparation)},
                    {"n_start_offsets", q.start_.size()},
                    {"n_dest_offsets", q.destination_.size()},
                    {"n_td_start_offsets", q.td_start_.size()},
                    {"n_td_dest_offsets", q.td_dest_.size()}};

    auto r = n::routing::routing_result{};
    auto algorithm = query.algorithm_;
    auto search_state = n::routing::search_state{};
    while (true) {
      if (algorithm == api::algorithmEnum::PONG && query.timetableView_ &&
          // arriveBy |  extend_later | PONG applicable
          // ---------+---------------+---------------------
          // FALSE    |  FALSE        | FALSE    => rRAPTOR
          // FALSE    |  TRUE         | TRUE     => PONG
          // TRUE     |  FALSE        | TRUE     => PONG
          // TRUE     |  TRUE         | FALSE    => rRAPTOR
          query.arriveBy_ != start_time.extend_interval_later_) {
        try {
          auto raptor_state = n::routing::raptor_state{};
          r = n::routing::pong_search(
              *tt_, rtt, search_state, raptor_state, q,
              query.arriveBy_ ? n::direction::kBackward
                              : n::direction::kForward,
              query.timeout_.has_value() ? std::chrono::seconds{*query.timeout_}
                                         : max_timeout);
        } catch (std::exception const& e) {
          std::cout << "PONG EXCEPTION: " << e.what() << "\n";
          algorithm = api::algorithmEnum::RAPTOR;
          continue;
        }
      } else if (algorithm == api::algorithmEnum::RAPTOR || tbd_ == nullptr ||
                 (rtt != nullptr && rtt->n_rt_transports() != 0U) ||
                 query.arriveBy_ || q.prf_idx_ != tbd_->prf_idx_ ||
                 q.allowed_claszes_ != n::routing::all_clasz_allowed() ||
                 !q.td_start_.empty() || !q.td_dest_.empty() ||
                 !q.transfer_time_settings_.default_ || !q.via_stops_.empty() ||
                 q.require_bike_transport_ || q.require_car_transport_) {
        auto raptor_state = n::routing::raptor_state{};
        r = n::routing::raptor_search(
            *tt_, rtt, search_state, raptor_state, q,
            query.arriveBy_ ? n::direction::kBackward : n::direction::kForward,
            query.timeout_.has_value() ? std::chrono::seconds{*query.timeout_}
                                       : max_timeout);
      } else {
        auto tb_state = n::routing::tb::query_state{*tt_, *tbd_};
        r = n::routing::tb::tb_search(*tt_, search_state, tb_state, q);
      }
      break;
    }

    metrics_->routing_journeys_found_.Increment(
        static_cast<double>(r.journeys_->size()));
    metrics_->routing_execution_duration_seconds_total_.Observe(
        static_cast<double>(r.search_stats_.execute_time_.count()) / 1000.0);

    if (!r.journeys_->empty()) {
      metrics_->routing_journey_duration_seconds_.Observe(static_cast<double>(
          to_seconds(r.journeys_->begin()->arrival_time() -
                     r.journeys_->begin()->departure_time())));
    }

    auto journeys = r.journeys_->els_;
    auto search_interval = r.interval_;
    if (query.maxItineraries_.has_value()) {
      search_interval = shrink(start_time.extend_interval_earlier_,
                               static_cast<std::size_t>(*query.maxItineraries_),
                               r.interval_, journeys);
    }

    direct_filter(direct, journeys);

    return {
        .debugOutput_ =
            join(std::move(prepare_stats), std::move(query_stats),
                 r.search_stats_.to_map(), std::move(r.algo_stats_)),
        .from_ = from_p,
        .to_ = to_p,
        .direct_ = std::move(direct),
        .itineraries_ = utl::to_vec(
            journeys,
            [&, cache = street_routing_cache_t{}](auto&& j) mutable {
              return journey_to_response(
                  w_, l_, pl_, *tt_, *tags_, fa_, e, rtt, matches_, elevations_,
                  shapes_, gbfs_rd, ae_, tz_, j, start, dest, cache,
                  blocked.get(),
                  query.requireCarTransport_ && query.useRoutedTransfers_,
                  osr_params, query.pedestrianProfile_, query.elevationCosts_,
                  query.joinInterlinedLegs_, query.detailedTransfers_,
                  query.withFares_, query.withScheduledSkippedStops_,
                  config_.timetable_.value().max_matching_distance_,
                  query.maxMatchingDistance_, api_version,
                  query.ignorePreTransitRentalReturnConstraints_,
                  query.ignorePostTransitRentalReturnConstraints_,
                  query.language_);
            }),
        .previousPageCursor_ =
            fmt::format("EARLIER|{}", to_seconds(search_interval.from_)),
        .nextPageCursor_ =
            fmt::format("LATER|{}", to_seconds(search_interval.to_)),
    };
  }

  return {
      .from_ = to_place(tt_, tags_, w_, pl_, matches_, ae_, tz_, lang, from),
      .to_ = to_place(tt_, tags_, w_, pl_, matches_, ae_, tz_, lang, to),
      .direct_ = std::move(direct),
      .itineraries_ = {}};
}

}  // namespace motis::ep
