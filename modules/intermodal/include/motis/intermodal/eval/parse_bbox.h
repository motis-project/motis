#pragma once

#include <memory>
#include <string>

#include "motis/intermodal/eval/bbox.h"

namespace motis::intermodal::eval {

std::unique_ptr<bbox> parse_bbox(std::string const& input);

}  // namespace motis::intermodal::eval
