#include "motis/timetable/service_date.h"

#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/timetable.h"

namespace n = nigiri;

namespace motis {

// TODO: go to first trip stop, not first transport stop (-> recalculate offset)
std::string get_service_date(n::timetable const& tt,
                             n::transport const t,
                             n::stop_idx_t const stop_idx) {
  auto const o = tt.transport_first_dep_offset_[t.t_idx_];
  auto const utc_dep =
      tt.event_mam(t.t_idx_, stop_idx, n::event_type::kDep).as_duration();
  auto const gtfs_static_dep = utc_dep + o;
  auto const [day_offset, tz_offset_minutes] =
      n::rt::split_rounded(gtfs_static_dep - utc_dep);
  auto const day = (tt.internal_interval_days().from_ +
                    std::chrono::days{to_idx(t.day_)} - day_offset);
  return fmt::format("{:%Y-%m-%d}", day);
}

}  // namespace motis