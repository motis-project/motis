#include "./events.h"

#include <algorithm>
#include <iterator>

#include "nigiri/timetable.h"

#include "utl/verify.h"

#include "./random.h"

namespace n = nigiri;
namespace po = boost::program_options;

namespace motis {

void event_sampler::add_options(po::options_description& desc) {
  desc.add_options()  //
      ("event_weighted", po::value(&enabled_)->default_value(enabled_),
       "sample from_place with probability proportional to its departure "
       "count and to_place with probability proportional to its arrival "
       "count within [first_day, last_day); with lb_rank/geo_rank, the "
       "weighting is applied within the respective rank bucket instead of "
       "picking the exact rank");
}

void event_sampler::verify(bool const use_population) const {
  utl::verify(!enabled_ || !use_population,
              "--event_weighted cannot be combined with --population_from/"
              "--population_to: both decide how `from` and `to` are picked");
}

void event_sampler::count_events(n::timetable const& tt,
                                 std::vector<n::location_idx_t> const& stops,
                                 date::sys_days const first_day,
                                 date::sys_days const last_day) {
  if (!enabled_) {
    return;
  }

  dep_weight_.assign(tt.n_locations(), 0.0);
  arr_weight_.assign(tt.n_locations(), 0.0);

  auto const day_from = tt.day_idx(first_day);
  auto const day_to = tt.day_idx(last_day);

  for (auto i = 0U; i != tt.n_routes(); ++i) {
    auto const route = n::route_idx_t{i};
    auto const loc_seq = tt.route_location_seq_[route];
    auto const n_stops = static_cast<n::stop_idx_t>(loc_seq.size());
    if (n_stops < 2U) {
      continue;
    }

    auto active_days = 0.0;
    for (auto const t : tt.route_transport_ranges_[route]) {
      for (auto day = day_from; day != day_to; ++day) {
        if (tt.is_transport_active(t, day)) {
          ++active_days;
        }
      }
    }
    if (active_days == 0.0) {
      continue;
    }

    for (auto stop_idx = n::stop_idx_t{0U}; stop_idx != n_stops; ++stop_idx) {
      auto const st = n::stop{loc_seq[stop_idx]};
      auto const l = to_idx(st.location_idx());
      if (stop_idx + 1U != n_stops && st.in_allowed()) {
        dep_weight_[l] += active_days;
      }
      if (stop_idx != 0U && st.out_allowed()) {
        arr_weight_[l] += active_days;
      }
    }
  }

  // stops itself is never reordered
  stops_ = stops;
  dep_cum_.reserve(stops_.size());
  auto acc = 0.0;
  for (auto const l : stops_) {
    acc += dep_weight_[to_idx(l)];
    dep_cum_.push_back(acc);
  }
}

n::location_idx_t event_sampler::random_from(
    n::timetable const& tt, std::vector<n::location_idx_t> const& stops) const {
  if (dep_cum_.empty() || dep_cum_.back() <= 0.0) {
    return random_stop(tt, stops);
  }
  auto const x = dep_cum_.back() *
                 (static_cast<double>(rand_in(0U, 1'000'000U)) / 1'000'000.0);
  auto const it = std::upper_bound(dep_cum_.begin(), dep_cum_.end(), x);
  auto const idx =
      std::min(static_cast<std::size_t>(std::distance(dep_cum_.begin(), it)),
               stops_.size() - 1U);
  return stops_[idx];
}

n::location_idx_t event_sampler::random_to(
    std::vector<n::location_idx_t> const& stops,
    std::size_t lo,
    std::size_t hi) const {
  hi = std::min(hi, stops.size());
  lo = std::min(lo, hi > 0U ? hi - 1U : 0U);
  auto total = 0.0;
  for (auto k = lo; k != hi; ++k) {
    total += arr_weight_[to_idx(stops[k])];
  }
  if (total <= 0.0) {
    return stops[lo + rand_in(0U, static_cast<std::uint32_t>(hi - lo))];
  }
  auto const x =
      total * (static_cast<double>(rand_in(0U, 1'000'000U)) / 1'000'000.0);
  auto acc = 0.0;
  for (auto k = lo; k != hi; ++k) {
    acc += arr_weight_[to_idx(stops[k])];
    if (x <= acc) {
      return stops[k];
    }
  }
  return stops[hi - 1U];
}

}  // namespace motis
