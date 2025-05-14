#pragma once

#include "nigiri/routing/journey.h"

#include "motis/odm/mixer.h"

namespace motis {

std::vector<nigiri::routing::journey> read(std::string_view csv);

std::string write(std::vector<nigiri::routing::journey> const&);

}  // namespace motis