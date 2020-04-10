#pragma once

#if defined(MOTIS_USE_STD)

#include <string>

namespace mcd {

using std::string;

}  // namespace mcd

#else

#include "cista/containers/string.h"

#include "motis/data.h"

namespace mcd {

using motis::data::string;

}  // namespace mcd

#endif