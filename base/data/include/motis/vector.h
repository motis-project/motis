#pragma once

#if defined(MOTIS_USE_STD)

#include <vector>

#include "utl/to_vec.h"

namespace mcd {

using indexed_vector = std::vector;
using std::vector;
using utl::to_vec;

}  // namespace mcd

#else

#include "cista/containers/vector.h"

#include "motis/data.h"

namespace mcd {

using motis::data::indexed_vector;
using motis::data::to_vec;
using motis::data::vector;

}  // namespace mcd

#endif