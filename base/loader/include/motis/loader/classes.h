#pragma once

#include "motis/hash_map.h"
#include "motis/string.h"

#include "motis/core/schedule/connection.h"

namespace motis::loader {

mcd::hash_map<mcd::string, service_class> const& class_mapping();

}  // namespace motis::loader
