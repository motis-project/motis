#include "motis/endpoints/routing.h"

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
#include "motis/journey_to_response.h"
#include "motis/max_distance.h"
#include "motis/mode_to_profile.h"
#include "motis/parse_location.h"
#include "motis/street_routing.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/time_conv.h"
#include "motis/update_rtt_td_footpaths.h"

namespace n = nigiri;
using namespace std::chrono_literals;

namespace motis::ep {

constexpr auto const kInfinityDuration =
    n::duration_t{std::numeric_limits<n::duration_t::rep>::max()};

using td_offsets_t =
    n::hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::thread_specific_ptr<n::routing::search_state> search_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::thread_specific_ptr<n::routing::raptor_state> raptor_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static boost::thread_specific_ptr<osr::bitvec<osr::node_idx_t>> blocked;

place_t get_place(n::timetable const* tt,
                  tag_lookup const* tags,
                  std::string_view s) {
  if (auto const location = parse_location(s); location.has_value()) {
    return *location;
  }
  utl::verify(tt != nullptr && tags != nullptr,
              R"(could not parse location (no timetable loaded): "{}")", s);
  return tt_location{tags->get_location(*tt, s)};
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

td_offsets_t routing::get_td_offsets(elevators const& e,
                                     osr::location const& pos,
                                     osr::direction const dir,
                                     std::vector<api::ModeEnum> const& modes,
                                     bool const wheelchair,
                                     std::chrono::seconds const max) const {
  if (!w_ || !l_ || !pl_ || !tt_ || !loc_tree_ || !matches_) {
    return {};
  }

  auto ret = hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>{};
  for (auto const m : modes) {
    auto const profile = to_profile(m, wheelchair);

    if (profile != osr::search_profile::kWheelchair) {
      continue;  // handled by get_offsets
    }

    utl::equal_ranges_linear(
        get_td_footpaths(*w_, *l_, *pl_, *tt_, *loc_tree_, e, *matches_,
                         n::location_idx_t::invalid(), pos, dir, profile, max,
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

std::vector<n::routing::offset> routing::get_offsets(
    osr::location const& pos,
    osr::direction const dir,
    std::vector<api::ModeEnum> const& modes,
    bool const wheelchair,
    std::chrono::seconds const max,
    gbfs::gbfs_data const* gbfs) const {
  if (!loc_tree_ || !pl_ || !tt_ || !loc_tree_ || !matches_) {
    return {};
  }

  auto offsets = std::vector<n::routing::offset>{};
  auto ignore_walk = false;

  auto const handle_mode = [&](api::ModeEnum const m) {
    auto const profile = to_profile(m, wheelchair);

    if (rt_->e_ && profile == osr::search_profile::kWheelchair) {
      return;  // handled by get_td_offsets
    }

    auto const max_dist = get_max_distance(profile, max);

    auto const near_stops = loc_tree_->in_radius(pos.pos_, max_dist);
    auto const near_stop_locations =
        utl::to_vec(near_stops, [&](n::location_idx_t const l) {
          return osr::location{tt_->locations_.coordinates_[l],
                               pl_->get_level(*w_, (*matches_)[l])};
        });

    if (profile == osr::search_profile::kBikeSharing) {
      if (gbfs == nullptr) {
        return;
      }

      auto providers = hash_set<gbfs_provider_idx_t>{};
      gbfs->provider_rtree_.in_radius(
          pos.pos_, max_dist, [&](auto const pi) { providers.insert(pi); });

      for (auto const& pi : providers) {
        auto const& provider = gbfs->providers_.at(pi);
        auto const sharing = provider->get_sharing_data(w_->n_nodes());
        auto const paths =
            osr::route(*w_, *l_, profile, pos, near_stop_locations,
                       static_cast<osr::cost_t>(max.count()), dir,
                       kMaxMatchingDistance, nullptr, &sharing);
        ignore_walk = true;
        for (auto const [p, l] : utl::zip(paths, near_stops)) {
          if (p.has_value()) {
            offsets.emplace_back(l, n::duration_t{p->cost_ / 60},
                                 static_cast<n::transport_mode_id_t>(
                                     kGbfsTransportModeIdOffset + to_idx(pi)));
          }
        }
      }

    } else {
      auto const paths = osr::route(*w_, *l_, profile, pos, near_stop_locations,
                                    static_cast<osr::cost_t>(max.count()), dir,
                                    kMaxMatchingDistance, nullptr, nullptr);
      for (auto const [p, l] : utl::zip(paths, near_stops)) {
        if (p.has_value()) {
          offsets.emplace_back(l, n::duration_t{p->cost_ / 60},
                               static_cast<n::transport_mode_id_t>(profile));
        }
      }
    }
  };

  if (utl::find(modes, api::ModeEnum::BIKE_RENTAL) != end(modes)) {
    handle_mode(api::ModeEnum::BIKE_RENTAL);
  }

  for (auto const m : modes) {
    if (m == api::ModeEnum::BIKE_RENTAL) {
      continue;  // handled above
    }
    if (m == api::ModeEnum::WALK && ignore_walk) {
      continue;
    }
    handle_mode(m);
  }

  return offsets;
}

std::vector<api::ModeEnum> get_from_modes(
    std::vector<api::ModeEnum> const& modes) {
  auto ret = std::vector<api::ModeEnum>{};
  for (auto const& m : modes) {
    switch (m) {
      case api::ModeEnum::WALK: ret.emplace_back(api::ModeEnum::WALK); break;
      case api::ModeEnum::CAR_HAILING: [[fallthrough]];
      case api::ModeEnum::CAR: ret.emplace_back(api::ModeEnum::CAR); break;
      case api::ModeEnum::BIKE_TO_PARK: [[fallthrough]];
      case api::ModeEnum::BIKE: ret.emplace_back(api::ModeEnum::BIKE); break;
      case api::ModeEnum::BIKE_RENTAL:
        ret.emplace_back(api::ModeEnum::BIKE_RENTAL);
        break;

      case api::ModeEnum::CAR_TO_PARK:
      case api::ModeEnum::CAR_SHARING:
      case api::ModeEnum::FLEXIBLE:
      case api::ModeEnum::CAR_RENTAL:
      case api::ModeEnum::SCOOTER_RENTAL:
      case api::ModeEnum::CAR_PICKUP: throw utl::fail("mode not supported yet");

      default: continue;
    }
  }
  return ret;
}

std::vector<api::ModeEnum> get_to_modes(
    std::vector<api::ModeEnum> const& modes) {
  auto ret = std::vector<api::ModeEnum>{};
  for (auto const& m : modes) {
    switch (m) {
      case api::ModeEnum::WALK: ret.emplace_back(api::ModeEnum::WALK); break;
      case api::ModeEnum::BIKE: ret.emplace_back(api::ModeEnum::BIKE); break;
      case api::ModeEnum::BIKE_RENTAL:
        ret.emplace_back(api::ModeEnum::BIKE_RENTAL);
        break;

      case api::ModeEnum::CAR_TO_PARK:
      case api::ModeEnum::CAR_HAILING:
      case api::ModeEnum::CAR_SHARING:
      case api::ModeEnum::CAR_RENTAL:
      case api::ModeEnum::FLEXIBLE:
      case api::ModeEnum::SCOOTER_RENTAL:
      case api::ModeEnum::CAR_PICKUP: throw utl::fail("mode not supported yet");

      default: continue;
    }
  }
  return ret;
}

n::routing::clasz_mask_t to_clasz_mask(std::vector<api::ModeEnum> const& mode) {
  auto mask = n::routing::clasz_mask_t{0U};
  auto const allow = [&](n::clasz const c) {
    mask |= (1U << static_cast<std::underlying_type_t<n::clasz>>(c));
  };
  for (auto const& m : mode) {
    switch (m) {
      case api::ModeEnum::TRANSIT:
        mask = n::routing::all_clasz_allowed();
        return mask;
      case api::ModeEnum::TRAM: allow(n::clasz::kTram); break;
      case api::ModeEnum::SUBWAY: allow(n::clasz::kSubway); break;
      case api::ModeEnum::FERRY: allow(n::clasz::kShip); break;
      case api::ModeEnum::AIRPLANE: allow(n::clasz::kAir); break;
      case api::ModeEnum::BUS: allow(n::clasz::kBus); break;
      case api::ModeEnum::COACH: allow(n::clasz::kCoach); break;
      case api::ModeEnum::RAIL:
        allow(n::clasz::kHighSpeed);
        allow(n::clasz::kLongDistance);
        allow(n::clasz::kNight);
        allow(n::clasz::kRegional);
        allow(n::clasz::kRegionalFast);
        allow(n::clasz::kMetro);
        allow(n::clasz::kSubway);
        break;
      case api::ModeEnum::HIGHSPEED_RAIL: allow(n::clasz::kHighSpeed); break;
      case api::ModeEnum::LONG_DISTANCE: allow(n::clasz::kLongDistance); break;
      case api::ModeEnum::NIGHT_RAIL: allow(n::clasz::kNight); break;
      case api::ModeEnum::REGIONAL_FAST_RAIL:
        allow(n::clasz::kRegionalFast);
        break;
      case api::ModeEnum::REGIONAL_RAIL: allow(n::clasz::kRegional); break;
      default: continue;
    }
  }
  return mask;
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
    gbfs::gbfs_data const* gbfs,
    api::Place const& from,
    api::Place const& to,
    std::vector<api::ModeEnum> const& modes,
    n::unixtime_t const start_time,
    bool wheelchair,
    std::chrono::seconds max) const {
  if (!w_ || !l_) {
    return {};
  }
  auto const omit_walk =
      gbfs != nullptr &&
      utl::find(modes, api::ModeEnum::BIKE_RENTAL) != end(modes);
  auto fastest_direct = kInfinityDuration;
  auto cache = street_routing_cache_t{};
  auto itineraries = std::vector<api::Itinerary>{};
  for (auto const& m : modes) {
    if (m == api::ModeEnum::CAR || m == api::ModeEnum::BIKE ||
        (!omit_walk && m == api::ModeEnum::WALK)) {
      auto itinerary = route(
          *w_, *l_, gbfs, e, from, to, m, wheelchair, start_time, std::nullopt,
          gbfs_provider_idx_t::invalid(), cache, *blocked, max);
      if (itinerary.legs_.empty()) {
        continue;
      }
      auto const duration = std::chrono::duration_cast<n::duration_t>(
          std::chrono::seconds{itinerary.duration_});
      if (duration < fastest_direct) {
        fastest_direct = duration;
      }
      itineraries.emplace_back(std::move(itinerary));
    } else if (m == api::ModeEnum::BIKE_RENTAL && gbfs != nullptr) {
      auto const max_dist =
          get_max_distance(osr::search_profile::kBikeSharing, max);
      auto providers = hash_set<gbfs_provider_idx_t>{};
      gbfs->provider_rtree_.in_radius(
          {from.lat_, from.lon_}, max_dist,
          [&](auto const pi) { providers.insert(pi); });
      for (auto const& pi : providers) {
        auto itinerary =
            route(*w_, *l_, gbfs, e, from, to, m, wheelchair, start_time,
                  std::nullopt, pi, cache, *blocked, max);
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
  return {itineraries, fastest_direct};
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
      n::duration_t{std::numeric_limits<n::duration_t>::max()};

  auto const worse_than_fastest_direct = [&](n::duration_t const min) {
    return [&, min](auto const& o) {
      return o.duration() + min >= q.fastest_direct_;
    };
  };
  auto const get_min_duration = [&](auto&& x) {
    return x.empty() ? kMaxDuration
                     : utl::max_element(x, [](auto&& a, auto&& b) {
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

  utl::verify(min_start != kMaxDuration, "no valid start offset");
  utl::verify(min_dest != kMaxDuration, "no valid dest offset");

  utl::erase_if(q.start_, worse_than_fastest_direct(min_dest));
  utl::erase_if(q.destination_, worse_than_fastest_direct(min_start));
  for (auto& [k, v] : q.td_start_) {
    utl::erase_if(v, worse_than_fastest_direct(min_dest));
  }
  for (auto& [k, v] : q.td_dest_) {
    utl::erase_if(v, worse_than_fastest_direct(min_start));
  }
}

api::plan_response routing::operator()(boost::urls::url_view const& url) const {
  auto const rt = rt_;
  auto const rtt = rt->rtt_.get();
  auto const e = rt_->e_.get();
  auto const gbfs = gbfs_;
  if (blocked.get() == nullptr && w_ != nullptr) {
    blocked.reset(new osr::bitvec<osr::node_idx_t>{w_->n_nodes()});
  }

  auto const query = api::plan_params{url.params()};
  auto const modes = [&]() {
    auto m = query.mode_;
    utl::erase_duplicates(m);
    return m;
  }();
  auto const from = get_place(tt_, tags_, query.fromPlace_);
  auto const to = get_place(tt_, tags_, query.toPlace_);
  auto const from_p = to_place(tt_, tags_, w_, pl_, matches_, from);
  auto const to_p = to_place(tt_, tags_, w_, pl_, matches_, to);
  auto const from_modes = get_from_modes(modes);
  auto const to_modes = get_to_modes(modes);

  auto const& start = query.arriveBy_ ? to : from;
  auto const& dest = query.arriveBy_ ? from : to;
  auto const& start_modes = query.arriveBy_ ? to_modes : from_modes;
  auto const& dest_modes = query.arriveBy_ ? from_modes : to_modes;

  auto const [start_time, t] = get_start_time(query);

  UTL_START_TIMING(direct);
  auto const [direct, fastest_direct] =
      (holds_alternative<osr::location>(from) &&
       holds_alternative<osr::location>(to) && t.has_value())
          ? route_direct(e, gbfs.get(), from_p, to_p, from_modes, *t,
                         query.wheelchair_,
                         std::chrono::seconds{query.maxDirectTime_})
          : std::pair{std::vector<api::Itinerary>{}, kInfinityDuration};
  UTL_STOP_TIMING(direct);

  if (utl::find(modes, api::ModeEnum::TRANSIT) != end(modes) &&
      fastest_direct > 5min) {
    utl::verify(tt_ != nullptr && tags_ != nullptr,
                "mode=TRANSIT requires timetable to be loaded");

    UTL_START_TIMING(query_preparation);
    auto q = n::routing::query{
        .start_time_ = start_time.start_time_,
        .start_match_mode_ = get_match_mode(start),
        .dest_match_mode_ = get_match_mode(dest),
        .use_start_footpaths_ = !is_intermodal(start),
        .start_ = std::visit(
            utl::overloaded{
                [&](tt_location const l) { return station_start(l.l_); },
                [&](osr::location const& pos) {
                  auto const dir = query.arriveBy_ ? osr::direction::kBackward
                                                   : osr::direction::kForward;
                  return get_offsets(
                      pos, dir, start_modes, query.wheelchair_,
                      std::chrono::seconds{query.maxPreTransitTime_},
                      gbfs.get());
                }},
            start),
        .destination_ = std::visit(
            utl::overloaded{
                [&](tt_location const l) { return station_start(l.l_); },
                [&](osr::location const& pos) {
                  auto const dir = query.arriveBy_ ? osr::direction::kForward
                                                   : osr::direction::kBackward;
                  return get_offsets(
                      pos, dir, dest_modes, query.wheelchair_,
                      std::chrono::seconds{query.maxPostTransitTime_},
                      gbfs.get());
                }},
            dest),
        .td_start_ =
            rt_->e_ != nullptr
                ? std::visit(
                      utl::overloaded{
                          [&](tt_location) { return td_offsets_t{}; },
                          [&](osr::location const& pos) {
                            auto const dir = query.arriveBy_
                                                 ? osr::direction::kBackward
                                                 : osr::direction::kForward;
                            return get_td_offsets(
                                *e, pos, dir, start_modes, query.wheelchair_,
                                std::chrono::seconds{query.maxPreTransitTime_});
                          }},
                      start)
                : td_offsets_t{},
        .td_dest_ =
            rt_->e_ != nullptr
                ? std::visit(
                      utl::overloaded{
                          [&](tt_location) { return td_offsets_t{}; },
                          [&](osr::location const& pos) {
                            auto const dir = query.arriveBy_
                                                 ? osr::direction::kForward
                                                 : osr::direction::kBackward;
                            return get_td_offsets(
                                *e, pos, dir, dest_modes, query.wheelchair_,
                                std::chrono::seconds{
                                    query.maxPostTransitTime_});
                          }},
                      dest)
                : td_offsets_t{},
        .max_transfers_ = static_cast<std::uint8_t>(
            query.maxTransfers_.has_value() ? *query.maxTransfers_
                                            : n::routing::kMaxTransfers),
        .min_connection_count_ = static_cast<unsigned>(query.numItineraries_),
        .extend_interval_earlier_ = start_time.extend_interval_earlier_,
        .extend_interval_later_ = start_time.extend_interval_later_,
        .prf_idx_ = static_cast<n::profile_idx_t>(query.wheelchair_ ? 2U : 1U),
        .allowed_claszes_ = to_clasz_mask(modes),
        .require_bike_transport_ = query.requireBikeTransport_,
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

    auto const r = n::routing::raptor_search(
        *tt_, rtt, *search_state, *raptor_state, std::move(q),
        query.arriveBy_ ? n::direction::kBackward : n::direction::kForward,
        std::nullopt);

    return {
        .debugOutput_ = join(stats_map_t{{"direct", UTL_TIMING_MS(direct)},
                                         {"query_preparation",
                                          UTL_TIMING_MS(query_preparation)}},
                             r.search_stats_.to_map(), r.algo_stats_.to_map()),
        .from_ = from_p,
        .to_ = to_p,
        .direct_ = std::move(direct),
        .itineraries_ = utl::to_vec(
            *r.journeys_,
            [&, cache = street_routing_cache_t{}](auto&& j) mutable {
              return journey_to_response(w_, l_, pl_, *tt_, *tags_, e, rtt,
                                         matches_, shapes_, gbfs.get(),
                                         query.wheelchair_, j, start, dest,
                                         cache, *blocked);
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
