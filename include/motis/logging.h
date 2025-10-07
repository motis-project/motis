#pragma once

#include <string>
#include <string_view>

#include "motis/config.h"

namespace motis {

int set_log_level(std::string_view log_lvl);

int set_log_level(config const&);

int set_log_level(std::string&& log_lvl);

}  // namespace motis
