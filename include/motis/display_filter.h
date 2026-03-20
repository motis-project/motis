#pragma once

#include "nigiri/types.h"

namespace n = nigiri;
namespace motis {

int min_zoom_level(n::clasz const clasz, float const distance);
bool should_display(n::clasz const clasz,
                    int const zoom_level,
                    float const distance);

}  // namespace motis
