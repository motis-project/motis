#pragma once

#include "motis/odm/calibration/requirement.h"
#include "motis/odm/mix.h"

namespace motis::odm {

mixer read_parameters(std::string_view json);

std::vector<requirement> read_requirements(std::string_view json);

}  // namespace motis::odm