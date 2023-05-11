#pragma once

#include <string>

#include "motis/core/schedule/schedule.h"

#include "motis/module/json_format.h"

namespace motis::debug {

std::string to_fbs_json(
    schedule const& sched, motis::trip const* trp,
    motis::module::json_format jf = motis::module::json_format::SINGLE_LINE);

}  // namespace motis::debug
