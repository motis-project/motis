#pragma once

#if defined(MOTIS_USE_STD)

#include <memory>

namespace mcd {

using std::make_unique;
using std::vector;

}  // namespace mcd

#else

#include "cista/containers/unique_ptr.h"

#include "motis/data.h"

namespace mcd {

using motis::data::make_unique;
using motis::data::unique_ptr;

}  // namespace mcd

#endif