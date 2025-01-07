#pragma once

#include <string>

#include "motis/odm/prima_state.h"

namespace motis::odm {

std::string serialize(prima_state const&, n::timetable const&);

}  // namespace motis::odm