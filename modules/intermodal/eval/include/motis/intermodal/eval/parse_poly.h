#pragma once

#include <memory>
#include <string>

#include "motis/intermodal/eval/poly.h"

namespace motis::intermodal::eval {

std::unique_ptr<poly> parse_poly(std::string const& filename);

}  // namespace motis::intermodal::eval
