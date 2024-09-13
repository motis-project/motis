#include "icc/endpoints/routing.h"

#include "boost/thread/tss.hpp"

#include "osr/platforms.h"
#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"

#include "nigiri/routing/limits.h"
#include "nigiri/routing/pareto_set.h"
#include "nigiri/routing/query.h"
#include "nigiri/routing/raptor/raptor_state.h"
#include "nigiri/routing/raptor_search.h"
#include "nigiri/special_stations.h"

#include "icc/constants.h"
#include "icc/endpoints/routing.h"
#include "icc/journey_to_response.h"
#include "icc/parse_location.h"
#include "icc/update_rtt_td_footpaths.h"

namespace json = boost::json;
namespace n = nigiri;

namespace icc::ep {

using td_offsets_t =
    n::hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>;

template <typename T>
concept JSON = boost::json::has_value_to<T>::value && std::is_aggregate_v<T>;

template <JSON T>
std::string to_str(T const& t) {
  return boost::json::serialize(boost::json::value_from(t));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<n::routing::search_state> search_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<n::routing::raptor_state> raptor_state;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
boost::thread_specific_ptr<osr::bitvec<osr::node_idx_t>> blocked;

place_t to_place(n::timetable const& tt, std::string_view s) {
  if (auto const location = parse_location(s); location.has_value()) {
    return *location;
  }
  try {
    return tt.locations_.get(n::location_id{s, n::source_idx_t{}}).l_;
  } catch (...) {
    throw utl::fail("could not find place {}", s);
  }
}

bool is_intermodal(place_t const& p) {
  return std::holds_alternative<osr::location>(p);
}

n::routing::location_match_mode get_match_mode(place_t const& p) {
  return is_intermodal(p) ? n::routing::location_match_mode::kIntermodal
                          : n::routing::location_match_mode::kEquivalent;
}

osr::search_profile to_profile(api::ModeEnum const m, bool const wheelchair) {
  switch (m) {
    case api::ModeEnum::WALK:
      return wheelchair ? osr::search_profile::kWheelchair
                        : osr::search_profile::kFoot;
    case api::ModeEnum::BIKE: return osr::search_profile::kBike;
    case api::ModeEnum::CAR: return osr::search_profile::kCar;
    default: throw utl::fail("unsupported mode");
  }
}

std::vector<n::routing::offset> direct(n::location_idx_t const l) {
  return {{l, n::duration_t{0U}, 0U}};
}

bool require_bike_transport(std::vector<api::ModeEnum> const& mode) {
  return utl::any_of(
      mode, [](api::ModeEnum const m) { return m == api::ModeEnum::BIKE; });
}

double get_max_distance(osr::search_profile const profile,
                        std::chrono::seconds const t) {
  switch (profile) {
    case osr::search_profile::kWheelchair: return t.count() * 0.8;
    case osr::search_profile::kFoot: return t.count() * 1.1;
    case osr::search_profile::kBike: return t.count() * 2.8;
    case osr::search_profile::kCar:
    case osr::search_profile::kCarParking: [[fallthrough]];
    case osr::search_profile::kCarParkingWheelchair: return t.count() * 28.0;
  }
  std::unreachable();
}

td_offsets_t routing::get_td_offsets(elevators const& e,
                                     osr::location const& pos,
                                     osr::direction const dir,
                                     std::vector<api::ModeEnum> const& modes,
                                     bool const wheelchair,
                                     std::chrono::seconds const max) const {
  auto ret = hash_map<n::location_idx_t, std::vector<n::routing::td_offset>>{};
  for (auto const m : modes) {
    auto const profile = to_profile(m, wheelchair);

    if (profile != osr::search_profile::kWheelchair) {
      continue;  // handled by get_offsets
    }

    utl::equal_ranges_linear(
        get_td_footpaths(w_, l_, pl_, tt_, loc_tree_, e, matches_, pos, dir,
                         profile, *blocked),
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
    std::chrono::seconds const max) const {
  auto offsets = std::vector<n::routing::offset>{};
  for (auto const m : modes) {
    auto const profile = to_profile(m, wheelchair);

    if (profile == osr::search_profile::kWheelchair) {
      continue;  // handled by get_td_offsets
    }

    auto const near_stops =
        loc_tree_.in_radius(pos.pos_, get_max_distance(profile, max));
    auto const near_stop_locations =
        utl::to_vec(near_stops, [&](n::location_idx_t const l) {
          return osr::location{tt_.locations_.coordinates_[l],
                               pl_.get_level(w_, matches_[l])};
        });
    auto const paths = osr::route(w_, l_, profile, pos, near_stop_locations,
                                  max.count(), dir, kMaxMatchingDistance);
    for (auto const [p, l] : utl::zip(paths, near_stops)) {
      if (p.has_value()) {
        offsets.emplace_back(
            n::routing::offset{l, n::duration_t{p->cost_ / 60},
                               static_cast<n::transport_mode_id_t>(profile)});
      }
    }
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

      case api::ModeEnum::CAR_TO_PARK:
      case api::ModeEnum::CAR_SHARING:
      case api::ModeEnum::FLEXIBLE:
      case api::ModeEnum::CAR_RENTAL:
      case api::ModeEnum::BIKE_RENTAL:
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

      case api::ModeEnum::CAR_TO_PARK:
      case api::ModeEnum::CAR_HAILING:
      case api::ModeEnum::CAR_SHARING:
      case api::ModeEnum::CAR_RENTAL:
      case api::ModeEnum::FLEXIBLE:
      case api::ModeEnum::BIKE_RENTAL:
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

n::routing::query get_start_time(api::plan_params const& query) {
  if (query.pageCursor_.has_value()) {
    return parse_cursor(*query.pageCursor_);
  } else {
    auto const t = get_date_time(query.date_, query.time_);
    auto const window = std::chrono::duration_cast<n::duration_t>(
        std::chrono::seconds{query.searchWindow_ * (query.arriveBy_ ? -1 : 1)});
    return {.start_time_ = query.timetableView_
                               ? n::routing::start_time_t{n::interval{
                                     query.arriveBy_ ? t - window : t,
                                     query.arriveBy_ ? t : t + window}}
                               : n::routing::start_time_t{t},
            .extend_interval_earlier_ = query.arriveBy_,
            .extend_interval_later_ = !query.arriveBy_};
  }
}

api::plan_response routing::operator()(boost::urls::url_view const& url) const {
  auto const rt = rt_;
  auto const rtt = rt->rtt_.get();
  auto const e = rt_->e_.get();
  if (blocked.get() == nullptr) {
    blocked.reset(new osr::bitvec<osr::node_idx_t>{w_.n_nodes()});
  }

  auto const query = api::plan_params{url.params()};
  auto const from = to_place(tt_, query.fromPlace_);
  auto const to = to_place(tt_, query.toPlace_);
  auto const from_modes = get_from_modes(query.mode_);
  auto const to_modes = get_to_modes(query.mode_);

  auto const& start = query.arriveBy_ ? to : from;
  auto const& dest = query.arriveBy_ ? from : to;
  auto const& start_modes = query.arriveBy_ ? to_modes : from_modes;
  auto const& dest_modes = query.arriveBy_ ? from_modes : to_modes;

  auto const start_time = get_start_time(query);
  auto q = n::routing::query{
      .start_time_ = start_time.start_time_,
      .start_match_mode_ = get_match_mode(start),
      .dest_match_mode_ = get_match_mode(dest),
      .use_start_footpaths_ = !is_intermodal(start),
      .start_ = std::visit(
          utl::overloaded{[&](n::location_idx_t const l) { return direct(l); },
                          [&](osr::location const& pos) {
                            auto const dir = query.arriveBy_
                                                 ? osr::direction::kForward
                                                 : osr::direction::kBackward;
                            return get_offsets(
                                pos, dir, start_modes, query.wheelchair_,
                                std::chrono::seconds{query.maxPreTransitTime_});
                          }},
          start),
      .destination_ = std::visit(
          utl::overloaded{
              [&](n::location_idx_t const l) { return direct(l); },
              [&](osr::location const& pos) {
                auto const dir = query.arriveBy_ ? osr::direction::kBackward
                                                 : osr::direction::kForward;
                return get_offsets(
                    pos, dir, dest_modes, query.wheelchair_,
                    std::chrono::seconds{query.maxPostTransitTime_});
              }},
          dest),
      .td_start_ = std::visit(
          utl::overloaded{
              [&](n::location_idx_t const l) { return td_offsets_t{}; },
              [&](osr::location const& pos) {
                auto const dir = query.arriveBy_ ? osr::direction::kForward
                                                 : osr::direction::kBackward;
                return get_td_offsets(
                    *e, pos, dir, start_modes, query.wheelchair_,
                    std::chrono::seconds{query.maxPreTransitTime_});
              }},
          start),
      .td_dest_ = std::visit(
          utl::overloaded{
              [&](n::location_idx_t const l) { return td_offsets_t{}; },
              [&](osr::location const& pos) {
                auto const dir = query.arriveBy_ ? osr::direction::kBackward
                                                 : osr::direction::kForward;
                return get_td_offsets(
                    *e, pos, dir, dest_modes, query.wheelchair_,
                    std::chrono::seconds{query.maxPostTransitTime_});
              }},
          dest),
      .max_transfers_ = static_cast<std::uint8_t>(
          query.maxTransfers_.has_value() ? *query.maxTransfers_
                                          : n::routing::kMaxTransfers),
      .min_connection_count_ = static_cast<unsigned>(query.numItineraries_),
      .extend_interval_earlier_ = start_time.extend_interval_earlier_,
      .extend_interval_later_ = start_time.extend_interval_later_,
      .prf_idx_ = static_cast<n::profile_idx_t>(query.wheelchair_ ? 2U : 1U),
      .allowed_claszes_ = to_clasz_mask(query.mode_),
      .require_bike_transport_ = require_bike_transport(query.mode_)};

  if (search_state.get() == nullptr) {
    search_state.reset(new n::routing::search_state{});
  }
  if (raptor_state.get() == nullptr) {
    raptor_state.reset(new n::routing::raptor_state{});
  }

  UTL_START_TIMING(nigiri);
  auto const r = n::routing::raptor_search(
      tt_, rtt, *search_state, *raptor_state, std::move(q),
      query.arriveBy_ ? n::direction::kBackward : n::direction::kForward,
      std::nullopt);
  UTL_STOP_TIMING(nigiri);
  auto const nigiri_timing = UTL_TIMING_MS(nigiri);

  auto const to_place = [&](place_t const p, std::string_view name) {
    return std::visit(
        utl::overloaded{
            [&](osr::location const l) {
              return api::Place{
                  .name_ = std::string{name},
                  .lat_ = l.pos_.lat_,
                  .lon_ = l.pos_.lng_,
                  .vertexType_ = api::VertexTypeEnum::NORMAL,
              };
            },
            [&](n::location_idx_t const l) {
              auto const pos = tt_.locations_.coordinates_[l];
              return api::Place{
                  .name_ = std::string{tt_.locations_.names_[l].view()},
                  .stopId_ = std::string{tt_.locations_.ids_[l].view()},
                  .lat_ = pos.lat_,
                  .lon_ = pos.lng_};
            }},
        p);
  };

  return {
      .from_ = to_place(from, "Origin"),
      .to_ = to_place(to, "Destination"),
      .itineraries_ =
          utl::to_vec(*r.journeys_,
                      [&, cache = street_routing_cache_t{}](auto&& j) mutable {
                        return journey_to_response(
                            w_, l_, tt_, pl_, *e, rtt, matches_,
                            query.wheelchair_, j, start, dest, cache, *blocked);
                      }),
      .previousPageCursor_ = fmt::format(
          "EARLIER|{}", std::chrono::duration_cast<std::chrono::seconds>(
                            r.interval_.from_.time_since_epoch())
                            .count()),
      .nextPageCursor_ = fmt::format(
          "LATER|{}", std::chrono::duration_cast<std::chrono::seconds>(
                          r.interval_.to_.time_since_epoch())
                          .count()),
  };
}

}  // namespace icc::ep