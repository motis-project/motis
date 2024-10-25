#include "motis/journey_to_response.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <span>

#include "utl/concat.h"
#include "utl/enumerate.h"
#include "utl/equal_ranges_linear.h"
#include "utl/overloaded.h"
#include "utl/verify.h"

#include "osr/platforms.h"
#include "osr/routing/route.h"
#include "osr/routing/sharing_data.h"

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
#include "motis/timetable/time_conv.h"
#include "motis/update_rtt_td_footpaths.h"

namespace n = nigiri;

namespace motis {

tt_location::tt_location(nigiri::rt::run_stop const& stop)
    : l_{stop.get_location_idx()},
      scheduled_{stop.get_scheduled_location_idx()} {}

tt_location::tt_location(nigiri::location_idx_t const l,
                         nigiri::location_idx_t const scheduled)
    : l_{l},
      scheduled_{scheduled == n::location_idx_t::invalid() ? l : scheduled} {}

api::Place to_place(osr::location const l, std::string_view name) {
  return {
      .name_ = std::string{name},
      .lat_ = l.pos_.lat_,
      .lon_ = l.pos_.lng_,
      .vertexType_ = api::VertexTypeEnum::NORMAL,
  };
}

osr::level_t get_lvl(osr::ways const* w,
                     osr::platforms const* pl,
                     platform_matches_t const* matches,
                     n::location_idx_t const l) {
  return w && pl && matches ? pl->get_level(*w, (*matches)[l])
                            : osr::level_t::invalid();
}

double get_level(osr::ways const* w,
                 osr::platforms const* pl,
                 platform_matches_t const* matches,
                 n::location_idx_t const l) {
  return to_float(get_lvl(w, pl, matches, l));
}

api::Place to_place(n::timetable const& tt,
                    tag_lookup const& tags,
                    osr::ways const* w,
                    osr::platforms const* pl,
                    platform_matches_t const* matches,
                    place_t const l,
                    place_t const start,
                    place_t const dest,
                    std::string_view name) {
  auto const is_track = [&](n::location_idx_t const x) {
    auto const type = tt.locations_.types_.at(x);
    return (type == n::location_type::kGeneratedTrack ||
            type == n::location_type::kTrack);
  };

  auto const get_track = [&](n::location_idx_t const x) {
    return is_track(x)
               ? std::optional{std::string{tt.locations_.names_.at(x).view()}}
               : std::nullopt;
  };

  return std::visit(
      utl::overloaded{
          [&](osr::location const& l) { return to_place(l, name); },
          [&](tt_location const tt_l) -> api::Place {
            auto const l = tt_l.l_;
            if (l == n::get_special_station(n::special_station::kStart)) {
              return to_place(std::get<osr::location>(start), "START");
            } else if (l == n::get_special_station(n::special_station::kEnd)) {
              return to_place(std::get<osr::location>(dest), "END");
            } else {
              auto const pos = tt.locations_.coordinates_[l];
              auto const p =
                  is_track(tt_l.l_) ? tt.locations_.parents_.at(l) : l;
              return {.name_ = std::string{tt.locations_.names_[p].view()},
                      .stopId_ = tags.id(tt, l),
                      .lat_ = pos.lat_,
                      .lon_ = pos.lng_,
                      .level_ = get_level(w, pl, matches, l),
                      .scheduledTrack_ = get_track(tt_l.scheduled_),
                      .track_ = get_track(tt_l.l_),
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
    case osr::search_profile::kBikeSharing: return api::ModeEnum::BIKE_RENTAL;
  }
  std::unreachable();
}

api::ModeEnum to_mode(osr::mode const m) {
  switch (m) {
    case osr::mode::kFoot: [[fallthrough]];
    case osr::mode::kWheelchair: return api::ModeEnum::WALK;
    case osr::mode::kBike:
      return api::ModeEnum::BIKE;  // TODO: BIKE_RENTAL / SCOOTER_RENTAL...
    case osr::mode::kCar: return api::ModeEnum::CAR;
  }
  std::unreachable();
}

api::Leg& write_leg(api::Itinerary& itinerary,
                    api::ModeEnum const mode,
                    api::Place const& from,
                    api::Place const& to,
                    std::chrono::sys_seconds const dep_time,
                    std::chrono::sys_seconds const arr_time) {
  auto& leg = itinerary.legs_.emplace_back();
  leg.mode_ = mode;
  leg.from_ = from;
  leg.to_ = to;
  leg.from_.departure_ = leg.startTime_ = dep_time;
  leg.to_.arrival_ = leg.endTime_ = arr_time;
  leg.duration_ = (arr_time - dep_time).count();
  return leg;
}

api::Leg& write_leg(api::Itinerary& itinerary,
                    api::ModeEnum const mode,
                    n::timetable const& tt,
                    tag_lookup const& tags,
                    osr::ways const* w,
                    osr::platforms const* pl,
                    platform_matches_t const* matches,
                    n::routing::journey::leg const& j_leg,
                    place_t const& start,
                    place_t const& dest) {
  auto const pred = itinerary.legs_.empty() ? nullptr : &itinerary.legs_.back();

  return write_leg(
      itinerary, mode,
      pred == nullptr ? to_place(tt, tags, w, pl, matches,
                                 tt_location{j_leg.from_}, start, dest)
                      : pred->to_,
      to_place(tt, tags, w, pl, matches, tt_location{j_leg.to_}, start, dest),
      j_leg.dep_time_, j_leg.arr_time_);
}

api::Itinerary journey_to_response(osr::ways const* w,
                                   osr::lookup const* l,
                                   osr::platforms const* pl,
                                   n::timetable const& tt,
                                   tag_lookup const& tags,
                                   elevators const* e,
                                   n::rt_timetable const* rtt,
                                   platform_matches_t const* matches,
                                   n::shapes_storage const* shapes,
                                   gbfs::gbfs_data const* gbfs,
                                   bool const wheelchair,
                                   n::routing::journey const& j,
                                   place_t const& start,
                                   place_t const& dest,
                                   street_routing_cache_t& cache,
                                   osr::bitvec<osr::node_idx_t>& blocked_mem) {
  auto const to_location = [&](n::location_idx_t const loc) {
    switch (to_idx(loc)) {
      case static_cast<n::location_idx_t::value_t>(n::special_station::kStart):
        assert(std::holds_alternative<osr::location>(start));
        return std::get<osr::location>(start);
      case static_cast<n::location_idx_t::value_t>(n::special_station::kEnd):
        assert(std::holds_alternative<osr::location>(dest));
        return std::get<osr::location>(dest);
      default:
        return osr::location{tt.locations_.coordinates_[loc],
                             get_lvl(w, pl, matches, loc)};
    }
  };

  auto const get_path =
      [&](osr::location const& from, osr::location const& to,
          n::transport_mode_id_t const transport_mode,
          osr::search_profile const profile, n::unixtime_t const start_time,
          osr::sharing_data const* sharing) -> std::optional<osr::path> {
    auto const s = e ? get_states_at(*w, *l, *e, start_time, from.pos_)
                     : std::optional{std::pair<nodes_t, states_t>{}};
    auto const& [e_nodes, e_states] = *s;
    auto const key = std::tuple{from, to, transport_mode, e_states};
    auto const it = cache.find(key);
    auto const path =
        it != end(cache)
            ? it->second
            : osr::route(
                  *w, *l, profile, from, to, 3600, osr::direction::kForward,
                  kMaxMatchingDistance,
                  s ? &set_blocked(e_nodes, e_states, blocked_mem) : nullptr,
                  sharing);
    if (it == end(cache)) {
      cache.emplace(std::pair{key, path});
    }
    if (!path.has_value()) {
      if (it == end(cache)) {
        std::cout << "no path found: " << from << " -> " << to
                  << ", profile=" << to_str(profile) << std::endl;
      }
    }
    return path;
  };

  auto const get_steps = [&](auto const& segments,
                             std::function<geo::polyline(
                                 osr::path::segment const&)> get_polyline) {
    return utl::to_vec(segments, [&](osr::path::segment const& s) {
      auto const way_name = s.way_ == osr::way_idx_t::invalid()
                                ? osr::string_idx_t::invalid()
                                : w->way_names_[s.way_];
      auto const polyline = get_polyline(s);
      return api::StepInstruction{
          .relativeDirection_ = api::RelativeDirectionEnum::CONTINUE,  // TODO
          .absoluteDirection_ = api::AbsoluteDirectionEnum::NORTH,  // TODO
          .distance_ = static_cast<double>(s.dist_),
          .fromLevel_ = to_float(s.from_level_),
          .toLevel_ = to_float(s.to_level_),
          .osmWay_ = s.way_ == osr::way_idx_t ::invalid()
                         ? std::nullopt
                         : std::optional{static_cast<std::int64_t>(
                               to_idx(w->way_osm_idx_[s.way_]))},
          .polyline_ = {geo::encode_polyline<7>(polyline),
                        static_cast<std::int64_t>(polyline.size())},
          .streetName_ = way_name == osr::string_idx_t::invalid()
                             ? ""
                             : std::string{w->strings_[way_name].view()},
          .exit_ = {},  // TODO
          .stayOn_ = false,  // TODO
          .area_ = false  // TODO
      };
    });
  };

  auto const add_routed_polyline = [&](osr::search_profile const profile,
                                       osr::location const& from,
                                       osr::location const& to, api::Leg& leg) {
    if (!w || !l) {
      return;
    }

    auto const path = get_path(
        from, to, static_cast<n::transport_mode_id_t>(profile), profile,
        std::chrono::time_point_cast<n::i32_minutes>(*leg.startTime_), nullptr);

    if (!path.has_value()) {
      return;
    }

    leg.steps_ = get_steps(path->segments_, [](osr::path::segment const& seg) {
      return seg.polyline_;
    });

    auto concat = geo::polyline{};
    for (auto const& p : path->segments_) {
      utl::concat(concat, p.polyline_);
    }
    leg.distance_ = path->dist_;
    leg.legGeometry_.points_ = geo::encode_polyline<7>(concat);
    leg.legGeometry_.length_ = static_cast<std::int64_t>(concat.size());
  };

  auto const add_rental_legs = [&](n::transport_mode_id_t const transport_mode,
                                   osr::location const& from,
                                   osr::location const& to,
                                   n::routing::journey::leg const& j_leg,
                                   api::Itinerary& itinerary) {
    auto start_time =
        std::chrono::time_point_cast<std::chrono::seconds>(j_leg.dep_time_);

    utl::verify(gbfs != nullptr, "missing gbfs data");
    auto const provider_idx =
        gbfs::gbfs_provider_idx_t{transport_mode - kGbfsTransportModeIdOffset};
    auto const& provider = gbfs->providers_.at(to_idx(provider_idx));
    auto const sharing =
        osr::sharing_data{.start_allowed_ = provider.start_allowed_,
                          .end_allowed_ = provider.end_allowed_,
                          .through_allowed_ = provider.through_allowed_,
                          .additional_node_offset_ = w->n_nodes(),
                          .additional_edges_ = provider.additional_edges_};

    auto const get_node_pos = [&](osr::node_idx_t const n) -> geo::latlng {
      if (n == osr::node_idx_t::invalid()) {
        return {};
      } else if (to_idx(n) < sharing.additional_node_offset_) {
        return w->get_node_pos(n).as_latlng();
      } else {
        auto const& an = provider.additional_nodes_.at(
            to_idx(n) - sharing.additional_node_offset_);
        if (std::holds_alternative<gbfs::additional_node::station>(an.data_)) {
          return provider.stations_
              .at(std::get<gbfs::additional_node::station>(an.data_).id_)
              .info_.pos_;
        } else {
          return provider.vehicle_status_
              .at(std::get<gbfs::additional_node::vehicle>(an.data_).idx_)
              .pos_;
        }
      }
    };

    auto const get_polyline =
        [&](osr::path::segment const& seg) -> geo::polyline {
      if (!seg.polyline_.empty()) {
        return seg.polyline_;
      } else {
        return geo::polyline{get_node_pos(seg.from_), get_node_pos(seg.to_)};
      }
    };

    auto const path = get_path(
        from, to, transport_mode, osr::search_profile::kBikeSharing,
        std::chrono::time_point_cast<n::i32_minutes>(start_time), &sharing);

    if (!path.has_value()) {
      return;
    }

    auto last_place = to_place(tt, tags, w, pl, matches,
                               tt_location{j_leg.from_}, start, dest);

    auto rental = api::Rental{
        .systemId_ = provider.sys_info_.id_,
        .systemName_ = provider.sys_info_.name_,
        .url_ = provider.sys_info_.url_,
    };

    using it_t = std::vector<osr::path::segment>::const_iterator;
    auto t = start_time;
    utl::equal_ranges_linear(
        path->segments_,
        [](auto const& a, auto const& b) { return a.mode_ == b.mode_; },
        [&](it_t const& lb, it_t const& ub) {
          auto const range = std::span{lb, ub};
          auto const is_last_leg = ub == end(path->segments_);
          auto const is_bike_leg = lb->mode_ == osr::mode::kBike;

          auto next_place =
              is_last_leg
                  ? to_place(tt, tags, w, pl, matches, tt_location{j_leg.to_},
                             start, dest)
                  : api::Place{.name_ = provider.sys_info_.name_,
                               .lat_ = 0,
                               .lon_ = 0,
                               .vertexType_ = api::VertexTypeEnum::BIKESHARE};

          if (!is_last_leg) {
            auto const to_node = range.back().to_;
            auto const to_pos = get_node_pos(to_node);
            next_place.lat_ = to_pos.lat_;
            next_place.lon_ = to_pos.lng_;

            if (to_idx(to_node) >= sharing.additional_node_offset_) {
              auto const& an = provider.additional_nodes_.at(
                  to_idx(to_node) - sharing.additional_node_offset_);
              std::visit(
                  utl::overloaded{
                      [&](gbfs::additional_node::station const& s) {
                        auto const& st = provider.stations_.at(s.id_);
                        next_place.name_ = st.info_.name_;
                        rental.stationName_ = st.info_.name_;
                        rental.rentalUriAndroid_ =
                            st.info_.rental_uris_.android_;
                        rental.rentalUriIOS_ = st.info_.rental_uris_.ios_;
                        rental.rentalUriWeb_ = st.info_.rental_uris_.web_;
                      },
                      [&](gbfs::additional_node::vehicle const& v) {
                        auto const& vi = provider.vehicle_status_.at(v.idx_);
                        rental.rentalUriAndroid_ = vi.rental_uris_.android_;
                        rental.rentalUriIOS_ = vi.rental_uris_.ios_;
                        rental.rentalUriWeb_ = vi.rental_uris_.web_;
                      }},
                  an.data_);
            }
          }

          auto& leg = write_leg(itinerary,
                                lb->mode_ == osr::mode::kBike
                                    ? api::ModeEnum::BIKE_RENTAL
                                    : api::ModeEnum::WALK,
                                last_place, next_place, t, t);

          if (is_bike_leg) {
            leg.rental_ = rental;
          }

          leg.steps_ = get_steps(range, get_polyline);

          auto concat = geo::polyline{};
          auto dist = 0.0;
          for (auto const& p : range) {
            auto const polyline = get_polyline(p);
            utl::concat(concat, polyline);
            if (p.cost_ != osr::kInfeasible) {
              t += std::chrono::seconds{p.cost_};
              dist += p.dist_;
            }
          }
          leg.distance_ = dist;
          leg.legGeometry_.points_ = geo::encode_polyline<7>(concat);
          leg.legGeometry_.length_ = static_cast<std::int64_t>(concat.size());

          leg.to_.arrival_ = leg.endTime_ = t;
          leg.duration_ =
              std::chrono::duration_cast<std::chrono::seconds>(t - start_time)
                  .count();
          start_time = t;
          last_place = next_place;
        });
  };

  auto const add_osr_legs =
      [&](n::transport_mode_id_t const transport_mode,
          osr::location const& from, osr::location const& to,
          n::routing::journey::leg const& j_leg, api::Itinerary& itinerary) {
        if (transport_mode >= kGbfsTransportModeIdOffset) {
          return add_rental_legs(transport_mode, from, to, j_leg, itinerary);
        } else {
          auto& leg = write_leg(itinerary, api::ModeEnum::WALK, tt, tags, w, pl,
                                matches, j_leg, start, dest);
          add_routed_polyline(static_cast<osr::search_profile>(transport_mode),
                              from, to, leg);
        }
      };

  auto itinerary = api::Itinerary{
      .duration_ = to_seconds(j.arrival_time() - j.departure_time()),
      .startTime_ = j.legs_.front().dep_time_,
      .endTime_ = j.legs_.back().arr_time_,
      .transfers_ = std::max(
          static_cast<std::iterator_traits<
              decltype(j.legs_)::iterator>::difference_type>(0),
          utl::count_if(j.legs_, [](auto&& leg) {
            return holds_alternative<n::routing::journey::run_enter_exit>(
                leg.uses_);
          }) - 1)};

  for (auto const [_, j_leg] : utl::enumerate(j.legs_)) {
    std::visit(
        utl::overloaded{
            [&](n::routing::journey::run_enter_exit const& t) {
              // TODO interlining
              auto const fr = n::rt::frun{tt, rtt, t.r_};
              auto const enter_stop = fr[t.stop_range_.from_];
              auto const exit_stop = fr[t.stop_range_.to_ - 1U];
              auto const color = enter_stop.get_route_color();
              auto const agency = enter_stop.get_provider();

              auto& leg = write_leg(itinerary, api::ModeEnum::TRANSIT, tt, tags,
                                    w, pl, matches, j_leg, start, dest);
              leg.source_ = fmt::format("{}", fmt::streamed(fr.dbg()));
              leg.headsign_ = enter_stop.direction();
              leg.routeColor_ = to_str(color.color_);
              leg.routeTextColor_ = to_str(color.text_color_);
              leg.mode_ = to_mode(enter_stop.get_clasz());
              leg.realTime_ = fr.is_rt();
              leg.tripId_ = tags.id(tt, enter_stop);
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

              leg.from_ = to_place(tt, tags, w, pl, matches,
                                   tt_location{fr[t.stop_range_.from_]});
              leg.to_ = to_place(tt, tags, w, pl, matches,
                                 tt_location{fr[t.stop_range_.to_ - 1U]});

              auto const first =
                  static_cast<n::stop_idx_t>(t.stop_range_.from_ + 1U);
              auto const last =
                  static_cast<n::stop_idx_t>(t.stop_range_.to_ - 1U);
              for (auto i = first; i < last; ++i) {
                auto const stop = fr[i];
                auto& p = leg.intermediateStops_->emplace_back(to_place(
                    tt, tags, w, pl, matches, tt_location{stop}, start, dest));
                p.departure_ = stop.time(n::event_type::kDep);
                p.departureDelay_ = to_ms(stop.delay(n::event_type::kDep));
                p.arrival_ = stop.time(n::event_type::kArr);
                p.arrivalDelay_ = to_ms(stop.delay(n::event_type::kArr));
              }
            },
            [&](n::footpath) {
              add_osr_legs(static_cast<n::transport_mode_id_t>(
                               wheelchair ? osr::search_profile::kWheelchair
                                          : osr::search_profile::kFoot),
                           to_location(j_leg.from_), to_location(j_leg.to_),
                           j_leg, itinerary);
            },
            [&](n::routing::offset const x) {
              add_osr_legs(x.transport_mode_id_, to_location(j_leg.from_),
                           to_location(j_leg.to_), j_leg, itinerary);
            }},
        j_leg.uses_);
  }

  return itinerary;
}
}  // namespace motis
