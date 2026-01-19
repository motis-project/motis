#include "motis/railviz.h"

#include <ranges>

#include "cista/containers/rtree.h"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "geo/box.h"
#include "geo/detail/register_box.h"
#include "geo/latlng.h"
#include "geo/polyline_format.h"

#include "nigiri/common/interval.h"
#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "motis/data.h"
#include "motis/journey_to_response.h"
#include "motis/parse_location.h"
#include "motis/tag_lookup.h"
#include "motis/timetable/clasz_to_mode.h"
#include "motis/timetable/time_conv.h"

namespace n = nigiri;

constexpr auto const kLimit = 12'000U;

using static_rtree = cista::raw::rtree<n::route_idx_t>;
using rt_rtree = cista::raw::rtree<n::rt_transport_idx_t>;

using int_clasz = decltype(n::kNumClasses);

namespace motis {

struct stop_pair {
  n::rt::run r_;
  n::stop_idx_t from_{}, to_{};
};

int min_zoom_level(n::clasz const clasz, float const distance) {
  switch (clasz) {
    // long distance
    case n::clasz::kAir:
    case n::clasz::kCoach:
      if (distance < 50'000.F) {
        return 8;  // typically long distance, maybe also quite short
      }
      [[fallthrough]];
    case n::clasz::kHighSpeed:
    case n::clasz::kLongDistance:
    case n::clasz::kNight: return 4;
    case n::clasz::kRegionalFast:
    case n::clasz::kRegional: return 7;

    // regional distance
    case n::clasz::kSuburban: return 8;

    // metro distance
    case n::clasz::kSubway: return 9;

    // short distance
    case n::clasz::kTram:
    case n::clasz::kBus: return distance > 10'000.F ? 9 : 10;

    // ship can be anything
    case n::clasz::kShip:
      if (distance > 100'000.F) {
        return 5;
      } else if (distance > 10'000.F) {
        return 8;
      } else {
        return 10;
      }

    case n::clasz::kODM:
    case n::clasz::kFunicular:
    case n::clasz::kAerialLift:
    case n::clasz::kOther: return 11;

    default: throw utl::fail("unknown n::clasz {}", static_cast<int>(clasz));
  }
}

bool should_display(n::clasz const clasz,
                    int const zoom_level,
                    float const distance) {
  return zoom_level >= min_zoom_level(clasz, distance);
}

struct route_geo_index {
  route_geo_index() = default;

  route_geo_index(n::timetable const& tt,
                  n::shapes_storage const* shapes_data,
                  n::clasz const clasz,
                  n::vector_map<n::route_idx_t, float>& distances) {
    // Fallback, if no route bounding box can be loaded
    auto const get_box = [&](n::route_idx_t const route_idx) {
      auto bounding_box = geo::box{};
      for (auto const l : tt.route_location_seq_[route_idx]) {
        bounding_box.extend(
            tt.locations_.coordinates_.at(n::stop{l}.location_idx()));
      }
      return bounding_box;
    };
    for (auto const [i, claszes] : utl::enumerate(tt.route_section_clasz_)) {
      auto const r = n::route_idx_t{i};
      if (claszes.at(0) != clasz) {
        continue;
      }
      auto const bounding_box = (shapes_data == nullptr)
                                    ? get_box(r)
                                    : shapes_data->get_bounding_box(r);

      rtree_.insert(bounding_box.min_.lnglat_float(),
                    bounding_box.max_.lnglat_float(), r);
      distances[r] = static_cast<float>(
          geo::distance(bounding_box.max_, bounding_box.min_));
    }
  }

  std::vector<n::route_idx_t> get_routes(geo::box const& b) const {
    auto routes = std::vector<n::route_idx_t>{};
    rtree_.search(b.min_.lnglat_float(), b.max_.lnglat_float(),
                  [&](auto, auto, n::route_idx_t const r) {
                    routes.push_back(r);
                    return true;
                  });
    return routes;
  }

  static_rtree rtree_{};
};

struct rt_transport_geo_index {
  rt_transport_geo_index() = default;

  rt_transport_geo_index(
      n::timetable const& tt,
      n::rt_timetable const& rtt,
      n::clasz const clasz,
      n::vector_map<n::rt_transport_idx_t, float>& distances) {
    for (auto const [i, claszes] :
         utl::enumerate(rtt.rt_transport_section_clasz_)) {
      auto const rt_t = n::rt_transport_idx_t{i};
      if (claszes.at(0) != clasz) {
        continue;
      }

      auto bounding_box = geo::box{};
      for (auto const l : rtt.rt_transport_location_seq_[rt_t]) {
        bounding_box.extend(
            tt.locations_.coordinates_.at(n::stop{l}.location_idx()));
      }

      rtree_.insert(bounding_box.min_.lnglat_float(),
                    bounding_box.max_.lnglat_float(), rt_t);
      distances[rt_t] = static_cast<float>(
          geo::distance(bounding_box.min_, bounding_box.max_));
    }
  }

  std::vector<n::rt_transport_idx_t> get_rt_transports(
      n::rt_timetable const& rtt, geo::box const& b) const {
    auto rt_transports = std::vector<n::rt_transport_idx_t>{};
    rtree_.search(b.min_.lnglat_float(), b.max_.lnglat_float(),
                  [&](auto, auto, n::rt_transport_idx_t const rt_t) {
                    if (!rtt.rt_transport_is_cancelled_[to_idx(rt_t)]) {
                      rt_transports.emplace_back(rt_t);
                    }
                    return true;
                  });
    return rt_transports;
  }

  rt_rtree rtree_{};
};

struct railviz_static_index::impl {
  std::array<route_geo_index, n::kNumClasses> static_geo_indices_;
  n::vector_map<n::route_idx_t, float> static_distances_{};
};

railviz_static_index::railviz_static_index(n::timetable const& tt,
                                           n::shapes_storage const* shapes_data)
    : impl_{std::make_unique<impl>()} {
  impl_->static_distances_.resize(tt.route_location_seq_.size());
  for (auto c = int_clasz{0U}; c != n::kNumClasses; ++c) {
    impl_->static_geo_indices_[c] =
        route_geo_index{tt, shapes_data, n::clasz{c}, impl_->static_distances_};
  }
}

railviz_static_index::~railviz_static_index() = default;

struct railviz_rt_index::impl {
  std::array<rt_transport_geo_index, n::kNumClasses> rt_geo_indices_;
  n::vector_map<n::rt_transport_idx_t, float> rt_distances_{};
};

railviz_rt_index::railviz_rt_index(nigiri::timetable const& tt,
                                   nigiri::rt_timetable const& rtt)
    : impl_{std::make_unique<impl>()} {
  impl_->rt_distances_.resize(rtt.rt_transport_location_seq_.size());
  for (auto c = int_clasz{0U}; c != n::kNumClasses; ++c) {
    impl_->rt_geo_indices_[c] =
        rt_transport_geo_index{tt, rtt, n::clasz{c}, impl_->rt_distances_};
  }
}

railviz_rt_index::~railviz_rt_index() = default;

void add_rt_transports(n::timetable const& tt,
                       n::rt_timetable const& rtt,
                       n::rt_transport_idx_t const rt_t,
                       n::interval<n::unixtime_t> const time_interval,
                       geo::box const& area,
                       std::vector<stop_pair>& runs) {
  auto const fr = n::rt::frun::from_rt(tt, &rtt, rt_t);
  for (auto const [from, to] : utl::pairwise(fr)) {
    auto const box = geo::make_box({from.pos(), to.pos()});
    if (!box.overlaps(area)) {
      continue;
    }

    auto const active =
        n::interval{from.time(n::event_type::kDep),
                    to.time(n::event_type::kArr) + n::i32_minutes{1}};
    if (active.overlaps(time_interval)) {
      utl::verify(runs.size() < kLimit, "too many trips");
      runs.emplace_back(
          stop_pair{.r_ = fr,  // NOLINT(cppcoreguidelines-slicing)
                    .from_ = from.stop_idx_,
                    .to_ = to.stop_idx_});
    }
  }
}

void add_static_transports(n::timetable const& tt,
                           n::rt_timetable const* rtt,
                           n::route_idx_t const r,
                           n::interval<n::unixtime_t> const time_interval,
                           geo::box const& area,
                           n::shapes_storage const* shapes_data,
                           std::vector<stop_pair>& runs) {
  auto const is_active = [&](n::transport const t) -> bool {
    return (rtt == nullptr
                ? tt.bitfields_[tt.transport_traffic_days_[t.t_idx_]]
                : rtt->bitfields_[rtt->transport_traffic_days_[t.t_idx_]])
        .test(to_idx(t.day_));
  };

  auto const seq = tt.route_location_seq_[r];
  auto const stop_indices =
      n::interval{n::stop_idx_t{0U}, static_cast<n::stop_idx_t>(seq.size())};
  auto const [start_day, _] = tt.day_idx_mam(time_interval.from_);
  auto const [end_day, _1] = tt.day_idx_mam(time_interval.to_);
  auto const get_box = [&](std::size_t segment) {
    if (shapes_data != nullptr) {
      auto const box = shapes_data->get_bounding_box(r, segment);
      if (box.has_value()) {
        return *box;
      }
    }
    // Fallback, if no segment bounding box can be loaded
    return geo::make_box(
        {tt.locations_.coordinates_[n::stop{seq[segment]}.location_idx()],
         tt.locations_.coordinates_[n::stop{seq[segment + 1]}.location_idx()]});
  };
  for (auto const [from, to] : utl::pairwise(stop_indices)) {
    // TODO dwell times
    auto const box = get_box(from);
    if (!box.overlaps(area)) {
      continue;
    }

    auto const arr_times = tt.event_times_at_stop(r, to, n::event_type::kArr);
    for (auto const [i, t_idx] :
         utl::enumerate(tt.route_transport_ranges_[r])) {
      auto const day_offset =
          static_cast<n::day_idx_t::value_t>(arr_times[i].days());
      for (auto traffic_day = start_day - day_offset; traffic_day <= end_day;
           ++traffic_day) {
        auto const t = n::transport{t_idx, traffic_day};
        if (time_interval.overlaps({tt.event_time(t, from, n::event_type::kDep),
                                    tt.event_time(t, to, n::event_type::kArr) +
                                        n::unixtime_t::duration{1}}) &&
            is_active(t)) {
          utl::verify(runs.size() < kLimit, "too many trips");
          runs.emplace_back(stop_pair{
              .r_ = n::rt::run{.t_ = t,
                               .stop_range_ = {from, static_cast<n::stop_idx_t>(
                                                         to + 1U)},
                               .rt_ = n::rt_transport_idx_t::invalid()},
              .from_ = 0,
              .to_ = 1});
        }
      }
    }
  }
}

api::trips_response get_trains(tag_lookup const& tags,
                               n::timetable const& tt,
                               n::rt_timetable const* rtt,
                               n::shapes_storage const* shapes,
                               osr::ways const* w,
                               osr::platforms const* pl,
                               platform_matches_t const* matches,
                               adr_ext const* ae,
                               tz_map_t const* tz,
                               railviz_static_index::impl const& static_index,
                               railviz_rt_index::impl const& rt_index,
                               api::trips_params const& query,
                               unsigned const api_version) {
  // Parse query.
  auto const zoom_level = static_cast<int>(query.zoom_);
  auto const min = parse_location(query.min_);
  auto const max = parse_location(query.max_);
  utl::verify(min.has_value(), "min not a coordinate: {}", query.min_);
  utl::verify(max.has_value(), "max not a coordinate: {}", query.max_);
  auto const start_time =
      std::chrono::time_point_cast<n::unixtime_t::duration>(*query.startTime_);
  auto const end_time =
      std::chrono::time_point_cast<n::unixtime_t::duration>(*query.endTime_);
  auto const time_interval = n::interval{start_time, end_time};
  auto const area = geo::make_box({min->pos_, max->pos_});

  // Collect runs within time+location window.
  auto runs = std::vector<stop_pair>{};
  for (auto c = int_clasz{0U}; c != n::kNumClasses; ++c) {
    auto const cl = n::clasz{c};
    if (!should_display(cl, zoom_level,
                        std::numeric_limits<float>::infinity())) {
      continue;
    }

    if (rtt != nullptr) {
      for (auto const& rt_t :
           rt_index.rt_geo_indices_[c].get_rt_transports(*rtt, area)) {
        if (should_display(cl, zoom_level, rt_index.rt_distances_[rt_t])) {
          add_rt_transports(tt, *rtt, rt_t, time_interval, area, runs);
        }
      }
    }

    for (auto const& r : static_index.static_geo_indices_[c].get_routes(area)) {
      if (should_display(cl, zoom_level, static_index.static_distances_[r])) {
        add_static_transports(tt, rtt, r, time_interval, area, shapes, runs);
      }
    }
  }

  // Create response.
  auto enc = geo::polyline_encoder<5>{};
  return utl::to_vec(runs, [&](stop_pair const& r) -> api::TripSegment {
    enc.reset();

    auto const fr = n::rt::frun{tt, rtt, r.r_};

    auto const from = fr[r.from_];
    auto const to = fr[r.to_];

    fr.for_each_shape_point(shapes,
                            {r.from_, static_cast<n::stop_idx_t>(r.to_ + 1U)},
                            [&](auto&& p) { enc.push(p); });

    return {
        .trips_ = {api::TripInfo{
            .tripId_ = tags.id(tt, from, n::event_type::kDep),
            .routeShortName_ =
                api_version < 4 ? std::optional{std::string{from.display_name(
                                      n::event_type::kDep, query.language_)}}
                                : std::nullopt,
            .displayName_ = api_version >= 4
                                ? std::optional{std::string{from.display_name(
                                      n::event_type::kDep, query.language_)}}
                                : std::nullopt}},
        .routeColor_ = to_str(from.get_route_color(n::event_type::kDep).color_),
        .mode_ = to_mode(from.get_clasz(n::event_type::kDep), api_version),
        .distance_ =
            fr.is_rt()
                ? rt_index.rt_distances_[fr.rt_]
                : static_index
                      .static_distances_[tt.transport_route_[fr.t_.t_idx_]],
        .from_ = to_place(&tt, &tags, w, pl, matches, ae, tz, query.language_,
                          tt_location{from}),
        .to_ = to_place(&tt, &tags, w, pl, matches, ae, tz, query.language_,
                        tt_location{to}),
        .departure_ = from.time(n::event_type::kDep),
        .arrival_ = to.time(n::event_type::kArr),
        .scheduledDeparture_ = from.scheduled_time(n::event_type::kDep),
        .scheduledArrival_ = to.scheduled_time(n::event_type::kArr),
        .realTime_ = fr.is_rt(),
        .polyline_ = std::move(enc.buf_)};
  });
}

}  // namespace motis