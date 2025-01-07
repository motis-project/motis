#pragma once

#include "nigiri/routing/journey.h"

namespace motis::odm {

bool operator==(nigiri::routing::journey const&,
                nigiri::routing::journey const&);

}  // namespace motis::odm