#include "motis/paxmon/update_capacity.h"

#include "motis/paxmon/trip_section_load_iterator.h"

namespace motis::paxmon {

void update_trip_capacity(universe& uv, schedule const& sched,
                          capacity_maps const& caps, trip const* trp,
                          bool const force_downgrade) {
  auto const sections = sections_with_load{sched, caps, uv, trp};
  if (!sections.has_paxmon_data() || sections.empty()) {
    return;
  }
  for (auto const& sec : sections) {
    auto const capacity = get_capacity(sched, sec.lcon(), sec.ev_key_from(),
                                       sec.ev_key_to(), caps);
    auto const new_source = capacity.second;
    // update capacity if new capacity source is better or equal (or forced)
    if (force_downgrade ||
        static_cast<std::underlying_type_t<capacity_source>>(new_source) <=
            static_cast<std::underlying_type_t<capacity_source>>(
                sec.get_capacity_source())) {
      auto const encoded_capacity = encode_capacity(capacity);
      auto* e = const_cast<edge*>(sec.paxmon_edge());  // NOLINT
      e->encoded_capacity_ = encoded_capacity;
    }
  }
}

}  // namespace motis::paxmon
