#pragma once

#if defined(MOTIS_USE_STD)

#include <array>

namespace mcd {

template <typename T, size_t Size>
using array = std::array<T, Size>;

}  // namespace mcd

#else

#include "cista/containers/array.h"

#include "motis/data.h"

namespace mcd {

using motis::data::array;

}  // namespace mcd

#endif