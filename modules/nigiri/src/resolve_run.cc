#include "motis/nigiri/resolve_run.h"

#include <algorithm>

#include "nigiri/rt/run.h"
#include "nigiri/timetable.h"

#include "motis/nigiri/location.h"
#include "motis/nigiri/unixtime_conv.h"

namespace n = nigiri;

namespace motis::nigiri {

n::rt::run resolve_run(tag_lookup const& tags, n::timetable const& tt,
                       extern_trip const& et) {
  auto const [tag, trip_id] = split_tag_and_location_id(et.id_);
  auto const src = tags.get_src(tag);
  auto const dep_time = to_nigiri_unixtime(et.time_);
  auto const day_idx = tt.day_idx(
      date::sys_days{std::chrono::time_point_cast<date::days>(dep_time)});

  auto const lb = std::lower_bound(
      begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_), trip_id,
      [&](n::pair<n::trip_id_idx_t, n::trip_idx_t> const& a,
          n::string const& b) {
        return std::tuple(tt.trip_id_src_[a.first],
                          tt.trip_id_strings_[a.first].view()) <
               std::tuple(src, std::string_view{b});
      });

  auto const id_matches = [src, trip_id = trip_id,
                           &tt](n::trip_id_idx_t const t_id_idx) {
    return tt.trip_id_src_[t_id_idx] == src &&
           tt.trip_id_strings_[t_id_idx].view() == trip_id;
  };

  for (auto i = lb; i != end(tt.trip_id_to_idx_) && id_matches(i->first); ++i) {
    for (auto const [t_idx, stop_range] :
         tt.trip_transport_ranges_[i->second]) {
      auto const day_offset =
          tt.event_mam(t_idx, stop_range.from_, n::event_type::kDep).days();
      auto const t = n::transport{t_idx, day_idx - day_offset};
      if (dep_time != tt.event_time(t, stop_range.from_, n::event_type::kDep)) {
        continue;
      }

      auto const& traffic_days =
          tt.bitfields_[tt.transport_traffic_days_[t_idx]];
      if (!traffic_days.test(to_idx(day_idx - day_offset))) {
        continue;
      }

      return n::rt::run{.t_ = t, .stop_range_ = stop_range};
    }
  }

  return {};
}

}  // namespace motis::nigiri