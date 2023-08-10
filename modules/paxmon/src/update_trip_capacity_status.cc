#include "motis/paxmon/update_trip_capacity_status.h"

#include "motis/paxmon/capacity_internal.h"
#include "motis/paxmon/trip_section_load_iterator.h"

namespace motis::paxmon {

void update_trip_capacity_status(schedule const& sched, universe& uv,
                                 trip const* trp, trip_data_index const tdi) {
  auto sections =
      sections_with_load{sched, uv, trp, tdi, capacity_info_source::TRIP};
  auto const section_count = sections.size();
  auto sections_with_capacity = 0U;
  auto worst_source = capacity_source::FORMATION_VEHICLES;

  for (auto const& sec : sections) {
    if (sec.has_capacity_info()) {
      ++sections_with_capacity;
      worst_source = get_worst_source(worst_source, sec.get_capacity_source());
    }
  }

  if (sections_with_capacity == 0) {
    worst_source = capacity_source::UNKNOWN;
  }

  auto& cap_status = uv.trip_data_.capacity_status(tdi);
  cap_status.has_trip_formation_ =
      get_trip_formation(uv.capacity_maps_, trp) != nullptr;
  cap_status.has_capacity_for_all_sections_ =
      (sections_with_capacity == section_count);
  cap_status.has_capacity_for_some_sections_ = (sections_with_capacity > 0);
  cap_status.worst_source_ = worst_source;
}

}  // namespace motis::paxmon
