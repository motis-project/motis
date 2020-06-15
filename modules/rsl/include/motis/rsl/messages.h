#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/rsl/combined_passenger_group.h"
#include "motis/rsl/rsl_data.h"
#include "motis/rsl/simulation_result.h"

#include "motis/module/message.h"

namespace motis::rsl {

motis::module::msg_ptr make_journeys_broken_msg(
    schedule const& sched, rsl_data const& data,
    std::map<unsigned, std::vector<combined_passenger_group>> const&
        combined_groups,
    simulation_result const& sim_result);

}  // namespace motis::rsl
