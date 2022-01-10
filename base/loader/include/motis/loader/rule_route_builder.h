#pragma once

#include <string>

#include "motis/loader/rule_service_graph_builder.h"

#include "motis/schedule-format/Schedule_generated.h"

namespace motis::loader {

void build_rule_routes(
    graph_builder& gb,
    flatbuffers64::Vector<flatbuffers64::Offset<RuleService>> const*
        rule_services);

}  // namespace motis::loader
