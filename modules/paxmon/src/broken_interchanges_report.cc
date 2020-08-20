#include "motis/paxmon/over_capacity_report.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>

#include "fmt/ostream.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/access/service_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/hash_map.h"

namespace motis::paxmon {

void write_broken_interchanges_report(paxmon_data const& data,
                                      std::string const& filename) {
  std::ofstream out{filename};
  auto const& g = data.graph_;

  out << "available,required\n";
  for (auto const& n : g.nodes_) {
    for (auto const& e : n->outgoing_edges(g)) {
      if (!e->is_interchange()) {
        continue;
      }
      auto const arrival = e->from(g)->current_time();
      auto const departure = e->to(g)->current_time();
      auto const ic_buffer =
          static_cast<int>(departure) - static_cast<int>(arrival);
      if (ic_buffer < e->transfer_time()) {
        out << ic_buffer << "," << e->transfer_time() << "\n";
      }
    }
  }
}

}  // namespace motis::paxmon
