#pragma once

#include <string>

#include "motis/odm/prima_state.h"

namespace motis::odm {

std::string serialize(prima_state const&, n::timetable const&);

void blacklist_update(prima_state&, std::string_view);

void whitelist_update(prima_state&, std::string_view);

}  // namespace motis::odm