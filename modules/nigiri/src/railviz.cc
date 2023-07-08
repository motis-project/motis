#include "motis/nigiri/railviz.h"

#include "boost/geometry/index/rtree.hpp"
#include "boost/iterator/function_output_iterator.hpp"

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/to_vec.h"

#include "geo/detail/register_box.h"
#include "geo/polyline_format.h"

#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"

#include "motis/core/conv/position_conv.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/nigiri/extern_trip.h"
#include "motis/nigiri/location.h"
#include "motis/nigiri/tag_lookup.h"
#include "motis/nigiri/unixtime_conv.h"

namespace n = nigiri;
namespace bgi = boost::geometry::index;
namespace mm = motis::module;
namespace fbs = flatbuffers;

using route_box = std::pair<geo::box, n::route_idx_t>;
using static_rtree = bgi::rtree<route_box, bgi::quadratic<16>>;

using rt_transport_box = std::pair<geo::box, n::rt_transport_idx_t>;
using rt_rtree = bgi::rtree<rt_transport_box, bgi::quadratic<16>>;

using int_clasz = decltype(n::kNumClasses);

namespace motis::nigiri {

struct route_geo_index {
  route_geo_index() = default;

  route_geo_index(n::timetable const& tt, n::clasz const clasz) {
    auto values = std::vector<route_box>{};
    for (auto const [r, claszes] : utl::enumerate(tt.route_section_clasz_)) {
      if (claszes.at(0) != clasz) {
        continue;
      }

      auto bounding_box = geo::box{};
      for (auto const l : tt.route_location_seq_[n::route_idx_t{r}]) {
        bounding_box.extend(
            tt.locations_.coordinates_.at(n::stop{l}.location_idx()));
      }

      values.emplace_back(bounding_box, n::route_idx_t{r});
    }
    rtree_ = static_rtree{values};
  }

  std::vector<n::route_idx_t> get_routes(geo::box const& b) const {
    std::vector<n::route_idx_t> routes;
    rtree_.query(bgi::intersects(b),
                 boost::make_function_output_iterator([&](route_box const& v) {
                   routes.emplace_back(v.second);
                 }));
    return routes;
  }

  static_rtree rtree_;
};

struct rt_transport_geo_index {
  rt_transport_geo_index() = default;

  rt_transport_geo_index(n::timetable const& tt, n::rt_timetable const& rtt,
                         n::clasz const clasz) {
    auto values = std::vector<rt_transport_box>{};
    for (auto const [rt_t, claszes] :
         utl::enumerate(rtt.rt_transport_section_clasz_)) {
      if (claszes.at(0) != clasz) {
        continue;
      }

      auto bounding_box = geo::box{};
      for (auto const l :
           rtt.rt_transport_location_seq_[n::rt_transport_idx_t{rt_t}]) {
        bounding_box.extend(
            tt.locations_.coordinates_.at(n::stop{l}.location_idx()));
      }

      values.emplace_back(bounding_box, n::rt_transport_idx_t{rt_t});
    }
  }

  std::vector<n::rt_transport_idx_t> get_rt_transports(
      geo::box const& b) const {
    std::vector<n::rt_transport_idx_t> routes;
    rtree_.query(bgi::intersects(b), boost::make_function_output_iterator(
                                         [&](rt_transport_box const& v) {
                                           routes.emplace_back(v.second);
                                         }));
    return routes;
  }

  rt_rtree rtree_;
};

struct railviz::impl {
  impl(tag_lookup const& tags, n::timetable const& tt) : tags_{tags}, tt_{tt} {
    for (auto c = int_clasz{0U}; c != n::kNumClasses; ++c) {
      static_geo_indices_[c] = route_geo_index{tt, n::clasz{c}};
    }
  }

  mm::msg_ptr get_trains(mm::msg_ptr const& msg) {
    using motis::railviz::RailVizTrainsRequest;
    auto const req = motis_content(RailVizTrainsRequest, msg);

    auto const start_time =
        n::unixtime_t{std::chrono::duration_cast<n::unixtime_t::duration>(
            std::chrono::seconds{req->start_time()})};
    auto const end_time =
        n::unixtime_t{std::chrono::duration_cast<n::unixtime_t::duration>(
            std::chrono::seconds{req->end_time()})};

    return get_trains(
        n::interval{start_time, end_time},
        geo::make_box({from_fbs(req->corner1()), from_fbs(req->corner2())}),
        req->max_trains(), req->zoom_geo());
  }

  mm::msg_ptr get_trains(n::interval<::nigiri::unixtime_t> time_interval,
                         geo::box const& area, int max_count, int zoom_level) {
    CISTA_UNUSED_PARAM(max_count)  // TODO(felix)
    CISTA_UNUSED_PARAM(zoom_level)  // TODO(felix)
    auto const [start_day, start_mam] = tt_.day_idx_mam(time_interval.from_);
    auto const [end_day, end_mam] = tt_.day_idx_mam(time_interval.to_);
    auto runs = std::vector<n::rt::run>{};
    for (auto c = int_clasz{0U}; c != n::kNumClasses; ++c) {
      for (auto const& rt_t : rt_geo_indices_[c].get_rt_transports(area)) {
        add_rt_transports(rt_t, time_interval, area, runs);
      }
      for (auto const& r : static_geo_indices_[c].get_routes(area)) {
        add_static_transports(r, time_interval, start_day, end_day, start_mam,
                              end_mam, area, runs);
      }
    }
    return create_response(runs);
  }

  mm::msg_ptr create_response(std::vector<n::rt::run> const& runs) const {
    geo::polyline_encoder<6> enc;

    mm::message_creator mc;

    auto stations = std::vector<fbs::Offset<Station>>{};
    auto known_stations = n::hash_set<n::location_idx_t>{};
    auto const add_station =
        [&](n::location_idx_t const l) -> n::location_idx_t {
      auto const x =
          tt_.locations_.types_[l] == n::location_type::kGeneratedTrack
              ? tt_.locations_.parents_[l]
              : l;
      if (known_stations.insert(x).second) {
        auto const pos = to_fbs(tt_.locations_.coordinates_[x]);
        stations.emplace_back(CreateStation(
            mc, mc.CreateString(get_station_id(tags_, tt_, x)),
            mc.CreateString(tt_.locations_.names_[x].view()), &pos));
      }
      return x;
    };

    auto polyline_indices_cache =
        n::hash_map<std::pair<n::location_idx_t, n::location_idx_t>,
                    std::int64_t>{};
    auto fbs_polylines = std::vector<fbs::Offset<fbs::String>>{
        mc.CreateString("") /* no zero, zero doesn't have a sign=direction */};
    auto const trains = utl::to_vec(runs, [&](n::rt::run const& r) {
      auto const fr = n::rt::frun{tt_, rtt_.get(), r};
      auto const from = fr[0];
      auto const to = fr[1];

      auto const from_l = add_station(from.get_location_idx());
      auto const to_l = add_station(to.get_location_idx());

      auto const key =
          std::pair{std::min(from_l, to_l), std::max(from_l, to_l)};
      auto const polyline_indices = std::vector<int64_t>{
          utl::get_or_create(
              polyline_indices_cache, key,
              [&] {
                enc.push(tt_.locations_.coordinates_.at(key.first));
                enc.push(tt_.locations_.coordinates_.at(key.second));
                fbs_polylines.emplace_back(mc.CreateString(enc.buf_));
                enc.reset();
                return static_cast<std::int64_t>(fbs_polylines.size() - 1U);
              }) *
          (key.first != from.get_location_idx() ? -1 : 1)};
      return motis::railviz::CreateTrain(
          mc, mc.CreateVector(std::vector{mc.CreateString(fr.name())}),
          static_cast<int>(fr.get_clasz()), get_route_distance(r),
          mc.CreateString(get_station_id(tags_, tt_, from_l)),
          mc.CreateString(get_station_id(tags_, tt_, to_l)),
          to_motis_unixtime(from.time(n::event_type::kDep)),
          to_motis_unixtime(from.time(n::event_type::kArr)),
          to_motis_unixtime(from.scheduled_time(n::event_type::kDep)),
          to_motis_unixtime(from.scheduled_time(n::event_type::kArr)),
          r.is_rt() ? TimestampReason_FORECAST : TimestampReason_SCHEDULE,
          r.is_rt() ? TimestampReason_FORECAST : TimestampReason_SCHEDULE,
          mc.CreateVector(std::vector{
              to_fbs(mc, nigiri_trip_to_extern_trip(tags_, tt_, fr.trip_idx(),
                                                    r.t_.day_))}),
          mc.CreateVector(polyline_indices));
    });

    auto extras = std::vector<std::uint64_t>{fbs_polylines.size() - 1};
    std::iota(begin(extras), end(extras), 1U);

    mc.create_and_finish(
        MsgContent_RailVizTrainsResponse,
        motis::railviz::CreateRailVizTrainsResponse(
            mc, mc.CreateVector(stations), mc.CreateVector(trains),
            mc.CreateVector(fbs_polylines), mc.CreateVector(extras))
            .Union());
    return mm::make_msg(mc);
  }

  double get_route_distance(n::rt::run const& r) const {
    if (r.is_rt()) {
      return 0.0;
    } else {
      return 0.0;
    }
  }

  void add_rt_transports(n::rt_transport_idx_t const rt_t,
                         n::interval<n::unixtime_t> const time_interval,
                         geo::box const& area, std::vector<n::rt::run>& runs) {
    auto const seq = rtt_->rt_transport_location_seq_[rt_t];
    auto const stop_indices =
        n::interval{n::stop_idx_t{0U}, static_cast<n::stop_idx_t>(seq.size())};
    for (auto const [from, to] : utl::pairwise(stop_indices)) {
      auto const box = geo::box{
          tt_.locations_.coordinates_[n::stop{seq[from]}.location_idx()],
          tt_.locations_.coordinates_[n::stop{seq[to]}.location_idx()]};
      if (!box.overlaps(area)) {
        continue;
      }

      auto const active =
          n::interval{rtt_->unix_event_time(rt_t, from, n::event_type::kDep),
                      rtt_->unix_event_time(rt_t, to, n::event_type::kArr) +
                          n::unixtime_t::duration{1U}};
      if (active.overlaps(time_interval)) {
        runs.emplace_back(n::rt::run{
            .stop_range_ = n::interval<n::stop_idx_t>{from, to}, .rt_ = rt_t});
      }
    }
  }

  void add_static_transports(n::route_idx_t const r,  //
                             n::interval<n::unixtime_t> const time_interval,
                             n::day_idx_t const start_day,
                             n::day_idx_t const end_day,
                             n::duration_t const start_mam,
                             n::duration_t const end_mam, geo::box const& area,
                             std::vector<n::rt::run>& runs) const {
    auto const is_active = [&](n::transport_idx_t const t,
                               n::day_idx_t const day) -> bool {
      return (rtt_ == nullptr
                  ? tt_.bitfields_[tt_.transport_traffic_days_[t]]
                  : rtt_->bitfields_[rtt_->transport_traffic_days_[t]])
          .test(to_idx(day));
    };

    auto const add_transports =
        [&](n::interval<n::stop_idx_t> const stop_range,
            n::interval<std::uint64_t> const transports) -> void {
      for (auto const idx_in_r : transports) {
        auto const t = tt_.route_transport_ranges_[r][idx_in_r];
        auto const ev =
            tt_.event_mam(r, t, stop_range.from_, n::event_type::kDep);
        for (auto const day : n::interval{start_day, end_day + 1}) {
          auto const active = n::interval{
              tt_.event_time(n::transport{t, day}, stop_range.from_,
                             n::event_type::kDep),
              tt_.event_time(n::transport{t, day}, stop_range.from_ + 1U,
                             n::event_type::kArr) +
                  n::unixtime_t::duration{1U}};
          if (time_interval.overlaps(active) && is_active(t, day - ev.days())) {
            runs.emplace_back(
                n::rt::run{.t_ = n::transport{t, day - ev.days()},
                           .stop_range_ = stop_range,
                           .rt_ = n::rt_transport_idx_t ::invalid()});
          }
        }
      }
    };

    auto const seq = tt_.route_location_seq_[r];
    auto const stop_indices =
        n::interval{n::stop_idx_t{0U}, static_cast<n::stop_idx_t>(seq.size())};
    for (auto const [from, to] : utl::pairwise(stop_indices)) {
      auto const stop_range =
          n::interval{from, static_cast<n::stop_idx_t>(to + 1)};
      auto const box = geo::box{
          tt_.locations_.coordinates_[n::stop{seq[from]}.location_idx()],
          tt_.locations_.coordinates_[n::stop{seq[to]}.location_idx()]};
      if (!box.overlaps(area)) {
        continue;
      }

      // Get first arrival after start_mam.
      auto const arr_times =
          tt_.event_times_at_stop(r, to, n::event_type::kArr);
      auto const first_arrival_after_start_mam =
          n::linear_lb(arr_times.begin(), arr_times.end(), start_mam,
                       [&](n::delta const a, n::duration_t const b) {
                         return a.mam() < b.count();
                       });
      auto const first_arr_after_start_mam_idx = static_cast<unsigned>(
          &*first_arrival_after_start_mam - &arr_times[0]);

      // Get last departure before end_mam.
      auto const dep_times =
          tt_.event_times_at_stop(r, from, n::event_type::kDep);
      auto const last_dep_before_end_mam =
          n::linear_lb(dep_times.rbegin(), dep_times.rend(), end_mam,
                       [&](n::delta const a, n::duration_t const b) {
                         return a.mam() > b.count();
                       });
      auto const last_dep_before_end_mam_idx =
          static_cast<unsigned>(&*last_dep_before_end_mam - &dep_times[0]);

      if (first_arr_after_start_mam_idx <= last_dep_before_end_mam_idx) {
        // Doesn't go over midnight, add continuous interval.
        add_transports(stop_range, n::interval{first_arr_after_start_mam_idx,
                                               last_dep_before_end_mam_idx});
      } else {
        // Goes over midnight, add intervals
        //   - [0, last_dep_before_end + 1[  -> next day
        //   - [first_arr_after_start, N[
        add_transports(stop_range,
                       n::interval{0U, last_dep_before_end_mam_idx + 1U});
        add_transports(stop_range, n ::interval{first_arr_after_start_mam_idx,
                                                arr_times.size()});
      }
    }
  }

  void update(std::shared_ptr<n::rt_timetable> const& rtt) {
    rtt_ = rtt;
    for (auto c = int_clasz{0U}; c != n::kNumClasses; ++c) {
      rt_geo_indices_[c] = rt_transport_geo_index{tt_, *rtt_, n::clasz{c}};
    }
  }

  tag_lookup const& tags_;
  n::timetable const& tt_;
  std::shared_ptr<n::rt_timetable> rtt_;
  std::array<route_geo_index, n::kNumClasses> static_geo_indices_;
  std::array<rt_transport_geo_index, n::kNumClasses> rt_geo_indices_;
};

railviz::railviz(tag_lookup const& tags, n::timetable const& tt)
    : impl_{std::make_unique<impl>(tags, tt)} {}

mm::msg_ptr railviz::get_trains(mm::msg_ptr const& msg) {
  return impl_->get_trains(msg);
}

void railviz::update(std::shared_ptr<n::rt_timetable> const& rtt) {
  impl_->update(rtt);
}

railviz::~railviz() = default;

}  // namespace motis::nigiri