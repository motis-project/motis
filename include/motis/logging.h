#pragma once

#include <string>

#include "motis/config.h"

namespace motis {

int set_log_level(config const&);

int set_log_level(std::string&& log_lvl);

}  // namespace motis
