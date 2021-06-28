#pragma once

#include <ctime>
#include <string>
#include <vector>

#include "motis/core/schedule/schedule.h"
#include "motis/module/message.h"

#include "motis/paxmon/tools/generator/search_interval_generator.h"

namespace motis::paxmon::tools::generator {

struct query_generator {
  query_generator(schedule const& sched);

  motis::module::msg_ptr get_routing_request(
      std::string const& target,
      motis::routing::Start start_type = motis::routing::Start_PretripStart,
      motis::routing::SearchDir dir = motis::routing::SearchDir_Forward);

  schedule const& sched_;
  search_interval_generator interval_gen_;
  std::vector<station_node const*> station_nodes_;
};

}  // namespace motis::paxmon::tools::generator
