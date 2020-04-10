#pragma once

#if defined(MOTIS_USE_STD)

#include <utility>

namespace mcd {

using std::pair;

}  // namespace mcd

#else

#include "cista/containers/pair.h"

#include "motis/data.h"

namespace mcd {

using motis::data::pair;

}  // namespace mcd

#endif