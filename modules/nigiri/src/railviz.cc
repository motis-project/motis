#include "motis/nigiri/railviz.h"

#include "utl/enumerate.h"

#include "boost/geometry/index/rtree.hpp"
#include "boost/iterator/function_output_iterator.hpp"

#include "geo/detail/register_box.h"

#include "nigiri/common/linear_lower_bound.h"
#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"

namespace n = nigiri;
namespace bgi = boost::geometry::index;

using value = std::pair<geo::box, n::route_idx_t>;
using rtree = bgi::rtree<value, bgi::quadratic<16>>;

using int_clasz = decltype(n::kNumClasses);

namespace motis::nigiri {

struct route_geo_index {
  route_geo_index() = default;

  route_geo_index(n::timetable const& tt, n::clasz const clasz) {
    auto values = std::vector<value>{};
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

    rtree_ = rtree{values};
  }

  std::vector<n::route_idx_t> get_routes(geo::box const& b) const {
    std::vector<n::route_idx_t> routes;
    rtree_.query(bgi::intersects(b),
                 boost::make_function_output_iterator(
                     [&](value const& v) { routes.emplace_back(v.second); }));
    return routes;
  }

  rtree rtree_;
};

struct train_retriever {
  train_retriever(n::timetable const& tt) : tt_{tt} {
    for (auto c = int_clasz{0U}; c != n::kNumClasses; ++c) {
      clasz_geo_index_[c] = route_geo_index{tt, n::clasz{c}};
    }
  }

  void get_trains(n::unixtime_t const start_time, n::unixtime_t const end_time,
                  int const max_count, int const last_count,
                  geo::box const& area, int const zoom_level) {
    auto const [start_day, start_mam] = tt_.day_idx_mam(start_time);
    auto const [end_day, end_mam] = tt_.day_idx_mam(end_time);
    auto runs = std::vector<n::rt::run>{};
    for (auto c = int_clasz{0U}; c != n::kNumClasses; ++c) {
      for (auto const& r : clasz_geo_index_[c].get_routes(area)) {
        add_trains(r, start_day, start_mam, end_mam, runs);
      }
    }
  }

  void add_trains(n::route_idx_t const r,  //
                  n::day_idx_t const day, n::duration_t const start_mam,
                  n::duration_t const end_mam,
                  std::vector<n::rt::run>& runs) const {
    for (auto stop_idx = n::stop_idx_t{0U};
         stop_idx != tt_.route_location_seq_[r].size(); ++stop_idx) {
      // Get first arrival after start_mam.
      auto const arr_times =
          tt_.event_times_at_stop(r, stop_idx, n::event_type::kDep);
      auto const first_arrival_after_start_mam =
          n::linear_lb(arr_times.begin(), arr_times.end(), start_mam,
                       [&](n::delta const a, n::duration_t const b) {
                         return a.mam() < b.count();
                       });
      auto const first_arrival_after_start_mam_idx = static_cast<unsigned>(
          &*first_arrival_after_start_mam - &arr_times[0]);

      // Get last departure after end_mam.
      auto const dep_times =
          tt_.event_times_at_stop(r, stop_idx, n::event_type::kDep);
      auto const last_departe_before_end_mam =
          n::linear_lb(dep_times.rbegin(), dep_times.rend(), start_mam,
                       [&](n::delta const a, n::duration_t const b) {
                         return a.mam() > b.count();
                       });
      auto const last_departe_before_end_mam_idx =
          static_cast<unsigned>(&*last_departe_before_end_mam - &dep_times[0]);

      if (first_arrival_after_start_mam_idx <=
          last_departe_before_end_mam_idx) {
        // Doesn't go over midnight, add continuous interval.
        add_transports(r, stop_idx, day,
                       n::interval{first_arrival_after_start_mam_idx,
                                   last_departe_before_end_mam_idx},
                       runs);
      } else {
        // Goes over midnight, add intervals
        //   - [0, last_dep_before_end + 1[
        //   - [first_arr_after_start, N[
        add_transports(r, stop_idx, day + 1U,
                       n::interval{0U, last_departe_before_end_mam_idx + 1U},
                       runs);
        add_transports(
            r, stop_idx, day,
            n ::interval{first_arrival_after_start_mam_idx, arr_times.size()},
            runs);
      }
    }
  }

  void add_transports(n::route_idx_t const r, n::stop_idx_t const stop,
                      n::day_idx_t const day,
                      n::interval<std::uint64_t> const transports,
                      std::vector<n::rt::run>& runs) const {
    for (auto const idx_in_r : transports) {
      auto const t = tt_.route_transport_ranges_[r][idx_in_r];
      if (is_active(t, day)) {
        runs.emplace_back(n::rt::run{
            .t_ = t,
            .stop_range_ =
                n::interval{stop, static_cast<n::stop_idx_t>(stop + 1U)},
            .rt_ = n::rt_transport_idx_t ::invalid()});
      }
    }
  }

  bool is_active(n::transport_idx_t const t, n::day_idx_t const day) const {
    return tt_.bitfields_[tt_.transport_traffic_days_[t]].test(to_idx(day));
  }

  n::timetable const& tt_;
  std::array<route_geo_index, n::kNumClasses> clasz_geo_index_;
};

}  // namespace motis::nigiri