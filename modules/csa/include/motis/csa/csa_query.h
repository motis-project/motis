#pragma once

#include <vector>

#include "motis/core/schedule/interval.h"
#include "motis/core/schedule/schedule.h"
#include "motis/csa/csa_timetable.h"

#include "motis/protocol/RoutingRequest_generated.h"

namespace motis::csa {

struct csa_query {
  csa_query(schedule const&, motis::routing::RoutingRequest const*);

  bool is_ontrip() const { return search_interval_.end_ == INVALID_TIME; }

  std::vector<station_id> meta_starts_, meta_dests_;
  interval search_interval_;
  unsigned min_connection_count_{0U};
  bool extend_interval_earlier_{false}, extend_interval_later_{false};
  search_dir dir_{search_dir::FWD};
  bool include_equivalent_{false};
};

}  // namespace motis::csa
