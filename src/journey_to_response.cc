#include "motis/journey_to_response.h"

#include <cmath>

#include "utl/concat.h"
#include "utl/enumerate.h"
#include "utl/overloaded.h"

#include "osr/platforms.h"
#include "osr/routing/route.h"

#include "geo/polyline_format.h"

#include "nigiri/common/split_duration.h"
#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/special_stations.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "motis/constants.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"
#include "motis/timetable/service_date.h"
#include "motis/timetable/time_conv.h"
#include "motis/update_rtt_td_footpaths.h"

namespace n = nigiri;

namespace motis {

api::Place to_place(osr::location const l, std::string_view name) {
  return {
      .name_ = std::string{name},
      .lat_ = l.pos_.lat_,
      .lon_ = l.pos_.lng_,
      .vertexType_ = api::VertexTypeEnum::NORMAL,
  };
}

api::Place to_place(n::timetable const& tt,
                    tag_lookup const& tags,
                    place_t const l,
                    place_t const start,
                    place_t const dest,
                    std::string_view name) {
  return std::visit(
      utl::overloaded{
          [&](osr::location const& l) { return to_place(l, name); },
          [&](n::location_idx_t const l) -> api::Place {
            if (l == n::get_special_station(n::special_station::kStart)) {
              return to_place(std::get<osr::location>(start), "START");
            } else if (l == n::get_special_station(n::special_station::kEnd)) {
              return to_place(std::get<osr::location>(dest), "END");
            } else {
              auto const pos = tt.locations_.coordinates_[l];
              auto const type = tt.locations_.types_.at(l);
              auto const is_track =
                  (type == n::location_type::kGeneratedTrack ||
                   type == n::location_type::kTrack);
              auto const p = is_track ? tt.locations_.parents_.at(l) : l;
              auto const track = is_track
                                     ? std::optional{std::string{
                                           tt.locations_.names_.at(l).view()}}
                                     : std::nullopt;
              return {.name_ = std::string{tt.locations_.names_[p].view()},
                      .stopId_ = tags.id(tt, l),
                      .lat_ = pos.lat_,
                      .lon_ = pos.lng_,
                      .track_ = track,
                      .vertexType_ = api::VertexTypeEnum::NORMAL};
            }
          }},
      l);
}

api::ModeEnum to_mode(osr::search_profile const m) {
  switch (m) {
    case osr::search_profile::kCarParkingWheelchair: [[fallthrough]];
    case osr::search_profile::kCarParking: return api::ModeEnum::CAR_TO_PARK;
    case osr::search_profile::kFoot: [[fallthrough]];
    case osr::search_profile::kWheelchair: return api::ModeEnum::WALK;
    case osr::search_profile::kCar: return api::ModeEnum::CAR;
    case osr::search_profile::kBike: return api::ModeEnum::BIKE;
  }
  std::unreachable();
}

api::Itinerary journey_to_response(
    osr::ways const& w,
    osr::lookup const& l,
    n::timetable const& tt,
    tag_lookup const& tags,
    osr::platforms const& pl,
    elevators const* e,
    n::rt_timetable const* rtt,
    vector_map<nigiri::location_idx_t, osr::platform_idx_t> const& matches,
    n::shapes_storage const* shapes,
    bool const wheelchair,
    n::routing::journey const& j,
    place_t const& start,
    place_t const& dest,
    street_routing_cache_t& cache,
    osr::bitvec<osr::node_idx_t>& blocked_mem) {
  auto const to_location = [&](n::location_idx_t const l) {
    switch (to_idx(l)) {
      case static_cast<n::location_idx_t::value_t>(n::special_station::kStart):
        assert(std::holds_alternative<osr::location>(start));
        return std::get<osr::location>(start);
      case static_cast<n::location_idx_t::value_t>(n::special_station::kEnd):
        assert(std::holds_alternative<osr::location>(dest));
        return std::get<osr::location>(dest);
      default:
        return osr::location{tt.locations_.coordinates_[l],
                             pl.get_level(w, matches[l])};
    }
  };
  auto const add_routed_polyline = [&](osr::search_profile const profile,
                                       osr::location const& from,
                                       osr::location const& to, api::Leg& leg) {
    auto const t = n::unixtime_t{std::chrono::duration_cast<n::i32_minutes>(
        std::chrono::milliseconds{leg.startTime_})};

    auto const s = e ? get_states_at(w, l, *e, t, from.pos_)
                     : std::optional{std::pair<nodes_t, states_t>{}};
    auto const& [e_nodes, e_states] = *s;
    auto const key = std::tuple{from, to, profile, e_states};
    auto const it = cache.find(key);
    auto const path =
        it != end(cache)
            ? it->second
            : osr::route(
                  w, l, profile, from, to, 3600, osr::direction::kForward,
                  kMaxMatchingDistance,
                  s ? &set_blocked(e_nodes, e_states, blocked_mem) : nullptr);
    if (it == end(cache)) {
      cache.emplace(std::pair{key, path});
    }

    if (!path.has_value()) {
      if (it == end(cache)) {
        std::cout << "no path found: " << from << " -> " << to
                  << ", profile=" << to_str(profile) << std::endl;
      }
      return;
    }

    leg.legGeometryWithLevels_ =
        utl::to_vec(path->segments_, [&](osr::path::segment const& s) {
          return api::LevelEncodedPolyline{
              .from_level_ = to_float(s.from_level_),
              .to_level_ = to_float(s.to_level_),
              .osm_way_ = s.way_ == osr::way_idx_t ::invalid()
                              ? std::nullopt
                              : std::optional{static_cast<std::int64_t>(
                                    to_idx(w.way_osm_idx_[s.way_]))},
              .polyline_ = {encode_polyline<7>(s.polyline_),
                            static_cast<std::int64_t>(s.polyline_.size())},
          };
        });

    auto concat = geo::polyline{};
    for (auto const& p : path->segments_) {
      utl::concat(concat, p.polyline_);
    }
    leg.distance_ = path->dist_;
    leg.legGeometry_.points_ = encode_polyline<7>(concat);
    leg.legGeometry_.length_ = static_cast<std::int64_t>(concat.size());
  };

  auto itinerary = api::Itinerary{
      .duration_ = to_seconds(j.arrival_time() - j.departure_time()),
      .startTime_ = to_ms(j.legs_.front().dep_time_),
      .endTime_ = to_ms(j.legs_.back().arr_time_),
      .transfers_ = std::max(
          static_cast<std::iterator_traits<
              decltype(j.legs_)::iterator>::difference_type>(0),
          utl::count_if(j.legs_, [](auto&& leg) {
            return holds_alternative<n::routing::journey::run_enter_exit>(
                leg.uses_);
          }) - 1)};

  for (auto const [_, j_leg] : utl::enumerate(j.legs_)) {
    auto const write_leg = [&](api::ModeEnum const mode) -> api::Leg& {
      auto& leg = itinerary.legs_.emplace_back();
      leg.mode_ = mode;
      leg.from_ = to_place(tt, tags, j_leg.from_, start, dest);
      leg.to_ = to_place(tt, tags, j_leg.to_, start, dest);
      leg.from_.departure_ = leg.startTime_ = to_ms(j_leg.dep_time_);
      leg.to_.arrival_ = leg.endTime_ = to_ms(j_leg.arr_time_);
      leg.duration_ = to_seconds(j_leg.arr_time_ - j_leg.dep_time_);
      return leg;
    };

    std::visit(
        utl::overloaded{
            [&](n::routing::journey::run_enter_exit const& t) {
              // TODO interlining
              auto const fr = n::rt::frun{tt, rtt, t.r_};
              auto const enter_stop = fr[t.stop_range_.from_];
              auto const exit_stop = fr[t.stop_range_.to_ - 1U];
              auto const color = enter_stop.get_route_color();
              auto const agency = enter_stop.get_provider();

              auto& leg = write_leg(api::ModeEnum::TRANSIT);
              leg.source_ = fmt::format("{}", fmt::streamed(fr.dbg()));
              leg.headsign_ = enter_stop.direction();
              leg.routeColor_ = to_str(color.color_);
              leg.routeTextColor_ = to_str(color.text_color_);
              leg.mode_ = to_mode(enter_stop.get_clasz());
              leg.realTime_ = fr.is_rt();
              leg.tripId_ =
                  fmt::format("{}_{}", tags.get_tag(fr.id().src_), fr.id().id_);
              leg.serviceDate_ = get_service_date(tt, t.r_.t_, 0U);
              leg.agencyName_ = agency.long_name_;
              leg.agencyId_ = agency.short_name_;
              leg.routeShortName_ = enter_stop.trip_display_name();
              leg.departureDelay_ =
                  to_ms(enter_stop.delay(n::event_type::kDep));
              leg.arrivalDelay_ = to_ms(exit_stop.delay(n::event_type::kArr));

              auto polyline = geo::polyline{};
              fr.for_each_shape_point(
                  shapes, t.stop_range_,
                  [&](geo::latlng const& pos) { polyline.emplace_back(pos); });
              leg.legGeometry_.points_ = geo::encode_polyline<7>(polyline);
              leg.legGeometry_.length_ =
                  static_cast<std::int64_t>(polyline.size());

              leg.intermediateStops_ = std::vector<api::Place>{};

              leg.from_.departureDelay_ = leg.departureDelay_ =
                  to_ms(fr[t.stop_range_.from_].delay(n::event_type::kDep));
              leg.to_.arrivalDelay_ = leg.arrivalDelay_ =
                  to_ms(fr[t.stop_range_.to_ - 1U].delay(n::event_type::kArr));

              auto const first =
                  static_cast<n::stop_idx_t>(t.stop_range_.from_ + 1U);
              auto const last =
                  static_cast<n::stop_idx_t>(t.stop_range_.to_ - 1U);
              for (auto i = first; i < last; ++i) {
                auto const stop = fr[i];
                auto& p = leg.intermediateStops_->emplace_back(
                    to_place(tt, tags, stop.get_location_idx(), start, dest));
                p.departure_ = to_ms(stop.time(n::event_type::kDep));
                p.departureDelay_ = to_ms(stop.delay(n::event_type::kDep));
                p.arrival_ = to_ms(stop.time(n::event_type::kArr));
                p.arrivalDelay_ = to_ms(stop.delay(n::event_type::kArr));
              }
            },
            [&](n::footpath) {
              auto& leg = write_leg(api::ModeEnum::WALK);
              add_routed_polyline(wheelchair ? osr::search_profile::kWheelchair
                                             : osr::search_profile::kFoot,
                                  to_location(j_leg.from_),
                                  to_location(j_leg.to_), leg);
            },
            [&](n::routing::offset const x) {
              auto const profile =
                  static_cast<osr::search_profile>(x.transport_mode_id_);
              auto& leg = write_leg(to_mode(profile));
              add_routed_polyline(profile, to_location(j_leg.from_),
                                  to_location(j_leg.to_), leg);
            }},
        j_leg.uses_);
  }

  return itinerary;
}

}  // namespace motis
