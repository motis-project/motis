#pragma once

#include "nigiri/routing/journey.h"

#include "motis/odm/mixer.h"

namespace motis::odm {

std::vector<nigiri::routing::journey> read(std::string_view csv);

std::string to_csv(std::vector<nigiri::routing::journey> const&);

}  // namespace motis::odm