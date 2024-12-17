#pragma once

#include <string>

#include "boost/json.hpp"

#include "motis/odm/prima.h"

namespace motis::odm {

std::string json_string(prima_state const&);

}  // namespace motis::odm