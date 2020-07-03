#pragma once

#include "motis/core/schedule/connection.h"

namespace motis::path {

int min_zoom_level(service_class clasz = service_class::OTHER,
                   float distance = 0.F);
bool should_display(service_class, int zoom_level, float distance = 0.F);

}  // namespace motis::path
