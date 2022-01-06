#pragma once

#if defined(MOTIS_USE_STD)

#include <variant>

namespace mcd {

using std::holds_alternative;
using std::variant;
using std::variant_size;

}  // namespace mcd

#else

#include "cista/containers/variant.h"

#include "motis/data.h"

namespace mcd {

using cista::get;
using cista::get_if;
using cista::holds_alternative;
using cista::variant;
using cista::variant_size;

}  // namespace mcd

#endif
