#include "motis/paxmon/sync_graph_load.h"

#ifdef MOTIS_CAPACITY_IN_SCHEDULE

#include "utl/verify.h"

#include "motis/paxmon/trip_section_load_iterator.h"

namespace motis::paxmon {

void sync_graph_load(schedule& sched, paxmon_data const& data,
                     trip const* trp) {
  for (auto const& ts : sections_with_load{sched, data, trp}) {
    auto& lc = const_cast<light_connection&>(ts.section_.lcon());  // NOLINT
    lc.capacity_ = ts.capacity();
    lc.passengers_ = ts.base_load();
  }
}

void sync_graph_load(schedule& sched, paxmon_data const& data) {
  for (auto const& trp : sched.trip_mem_) {
    sync_graph_load(sched, data, trp.get());
  }
}

void verify_graphs_synced(schedule const& sched, paxmon_data const& data) {
  for (auto const& trp : sched.trip_mem_) {
    for (auto const& ts : sections_with_load{sched, data, trp.get()}) {
      if (ts.edge_ != nullptr) {
        auto const& lc = ts.section_.lcon();
        utl::verify(lc.capacity_ == ts.edge_->capacity(),
                    "verify_graphs_synced: capacity mismatch: {} != {}",
                    lc.capacity_, ts.edge_->capacity());
        utl::verify(ts.base_load() == lc.passengers_,
                    "verify_graphs_synced: load mismatch: {} != {}",
                    ts.base_load(), lc.passengers_);
      }
    }
  }
}

}  // namespace motis::paxmon

#else

namespace motis::paxmon {

void sync_graph_load(schedule&, paxmon_data const&, trip const*) {}
void sync_graph_load(schedule&, paxmon_data const&) {}
void verify_graphs_synced(schedule const&, paxmon_data const&) {}

}  // namespace motis::paxmon

#endif
