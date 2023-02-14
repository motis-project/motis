#include "motis/paxmon/update_capacity.h"

#include "motis/core/access/trip_access.h"

#include "motis/paxmon/trip_section_load_iterator.h"

namespace motis::paxmon {

void update_trip_capacity(universe& uv, schedule const& sched,
                          trip const* trp) {
  auto const sections =
      sections_with_load{sched, uv, trp, capacity_info_source::LOOKUP};
  if (!sections.has_paxmon_data() || sections.empty()) {
    return;
  }
  for (auto const& sec : sections) {
    auto* e = const_cast<edge*>(sec.paxmon_edge());  // NOLINT
    e->encoded_capacity_ = sec.encoded_capacity();
  }
}

void update_all_trip_capacities(universe& uv, schedule const& sched) {
  for (auto const& [trp_idx, tdi] : uv.trip_data_.mapping_) {
    (void)tdi;
    auto const* trp = get_trip(sched, trp_idx);
    update_trip_capacity(uv, sched, trp);
  }
}

}  // namespace motis::paxmon
