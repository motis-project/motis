#pragma once

#include <string>

#include "boost/json.hpp"

#include "motis/odm/prima.h"

namespace motis::odm {

std::string serialize(prima_state const&);

void update(prima_state&, std::string_view);

}  // namespace motis::odm