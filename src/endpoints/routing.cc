#include "icc/endpoints/routing.h"

#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"

#include "nigiri/routing/query.h"

#include "icc/parse_location.h"

namespace json = boost::json;
namespace n = nigiri;

namespace icc::ep {

using place_t = std::variant<osr::location, n::location_idx_t>;

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
    case osr::search_profile::kCar: return t.count() * 28.0;
  }
  std::unreachable();
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
    auto const near_stops =
        rtree_.in_radius(pos.pos_, get_max_distance(profile, max));
    auto const near_stop_locations =
        utl::to_vec(near_stops, [&](n::location_idx_t const l) {
          return osr::location{tt_.locations_.coordinates_[l],
                               pl_.get_level(w_, matches_[l])};
        });
    auto const paths = osr::route(w_, l_, profile, pos, near_stop_locations,
                                  max.count(), dir, 8, nullptr);
    for (auto const [p, l] : utl::zip(paths, near_stops)) {
      if (p.has_value()) {
        offsets.emplace_back(
            n::routing::offset{l, n::duration_t{p->cost_ / 60},
                               static_cast<n::transport_mode_id_t>(m)});
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

api::plan_response routing::operator()(boost::urls::url_view const& url) const {
  auto const query = api::plan_params{url.params()};
  auto const from = to_place(tt_, query.fromPlace_);
  auto const to = to_place(tt_, query.toPlace_);
  auto const from_modes = get_from_modes(query.mode_);
  auto const to_modes = get_to_modes(query.mode_);
  auto const t = get_date_time(query.date_, query.time_);
  auto const window = std::chrono::duration_cast<n::duration_t>(
      std::chrono::seconds{query.searchWindow_ * (query.arriveBy_ ? -1 : 1)});
  auto const start_time = query.timetableView_
                              ? n::routing::start_time_t{n::interval{
                                    query.arriveBy_ ? t - window : t,
                                    query.arriveBy_ ? t : t + window}}
                              : n::routing::start_time_t{t};

  auto const& start = query.arriveBy_ ? to : from;
  auto const& dest = query.arriveBy_ ? from : to;
  auto const& start_modes = query.arriveBy_ ? to_modes : from_modes;
  auto const& dest_modes = query.arriveBy_ ? from_modes : to_modes;
  auto q = n::routing::query{
      .start_time_ = start_time,
      .start_match_mode_ = get_match_mode(start),
      .dest_match_mode_ = get_match_mode(dest),
      .use_start_footpaths_ = false,
      .start_ = std::visit(
          utl::overloaded{[&](n::location_idx_t const l) { return direct(l); },
                          [&](osr::location const& pos) {
                            auto const dir = query.arriveBy_
                                                 ? osr::direction::kBackward
                                                 : osr::direction::kForward;
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
          start),
      .max_transfers_ = static_cast<std::uint8_t>(
          query.maxTransfers_.has_value() ? *query.maxTransfers_
                                          : n::routing::kMaxTransfers),
      .min_connection_count_ = static_cast<unsigned>(query.numItineraries_),
      .extend_interval_earlier_ = true,
      .extend_interval_later_ = true,
      .prf_idx_ = static_cast<n::profile_idx_t>(query.wheelchair_ ? 2U : 1U),
      .allowed_claszes_ = to_clasz_mask(query.mode_),
      .require_bike_transport_ = require_bike_transport(query.mode_)};

  return api::plan_response{};
}

}  // namespace icc::ep