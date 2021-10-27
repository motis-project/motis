#include "motis/paxmon/over_capacity_report.h"

#include <fstream>
#include <iostream>

#include "motis/core/common/date_time_util.h"
#include "motis/core/access/trip_iterator.h"

namespace motis::paxmon {

void write_broken_interchanges_report(universe const& uv,
                                      std::string const& filename) {
  std::ofstream out{filename};

  out << "available,required\n";
  for (auto const& n : uv.graph_.nodes_) {
    for (auto const& e : n.outgoing_edges(uv)) {
      if (!e.is_interchange()) {
        continue;
      }
      auto const arrival = e.from(uv)->current_time();
      auto const departure = e.to(uv)->current_time();
      auto const ic_buffer =
          static_cast<int>(departure) - static_cast<int>(arrival);
      if (ic_buffer < e.transfer_time()) {
        out << ic_buffer << "," << e.transfer_time() << "\n";
      }
    }
  }
}

}  // namespace motis::paxmon
