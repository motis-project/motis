#include "motis/paxmon/update_capacity.h"

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

}  // namespace motis::paxmon
