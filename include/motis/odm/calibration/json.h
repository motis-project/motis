#pragma once

#include "motis/odm/calibration/requirement.h"

namespace motis::odm {

std::vector<requirement> read(std::string_view json);

}  // namespace motis::odm