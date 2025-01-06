#pragma once

#include "motis/odm/calibration/requirement.h"

namespace motis::odm {

std::vector<requirement> read_requirements(std::string_view json);

}  // namespace motis::odm