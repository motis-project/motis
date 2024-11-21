#include "motis/suspects.h"

#include "nigiri/timetable.h"

namespace n = nigiri;

namespace motis {

suspects::suspects(nigiri::timetable const& tt) {
  auto const is_logical = [](std::pair<geo::latlng, n::delta> const& dep,
                             std::pair<geo::latlng, n::delta> const& arr) {
    auto const [from_pos, dep_time] = dep;
    auto const [to_pos, arr_time] = arr;
    constexpr auto const kSlackBuffer = 5;
    auto const dist_in_m = geo::distance(from_pos, to_pos);
    if (dist_in_m <= 200.0) {
      return true;
    }
    auto const time_diff_in_min =
        arr_time.count() - dep_time.count() + kSlackBuffer;
    auto const speed_in_kmh = (dist_in_m / 1000.0) / (time_diff_in_min / 60.0);

    if (speed_in_kmh > 200) {
      std::cout << "time_diff=" << time_diff_in_min
                << ", dist_in_m=" << dist_in_m << ", speed=" << speed_in_kmh
                << "\n";
    }

    return speed_in_kmh < 200;
  };
  auto n_good_stop_pairs = 0U;
  for (auto r = n::route_idx_t{0U}; r != tt.n_routes(); ++r) {
    auto from_idx = n::stop_idx_t{0U}, to_idx = n::stop_idx_t{1U};
    for (auto const [a, b] : utl::pairwise(tt.route_location_seq_[r])) {
      auto const from = n::stop{a};
      auto const to = n::stop{b};
      for (auto const t : tt.route_transport_ranges_[r]) {
        auto const from_pos = tt.locations_.coordinates_[from.location_idx()];
        auto const to_pos = tt.locations_.coordinates_[to.location_idx()];
        auto const dep_time = tt.event_mam(r, t, from_idx, n::event_type::kDep);
        auto const arr_time = tt.event_mam(r, t, to_idx, n::event_type::kArr);
        if (!is_logical({from_pos, dep_time}, {to_pos, arr_time})) {
          routes_.push_back(r);
          goto next_route;
        }
        ++n_good_stop_pairs;
      }
      ++from_idx;
      ++to_idx;
    }
  next_route:;
  }
}

}  // namespace motis