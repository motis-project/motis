#pragma once

#include <cstdlib>
#include <functional>

#include "cista/equal_to.h"
#include "cista/hashing.h"

namespace motis {

template <typename HashFun, typename T>
struct deep_ptr_hash {
  std::size_t operator()(T const* arg) const {
    return arg == nullptr ? 0 : HashFun{}(*arg);
  }
};

template <typename T>
struct deep_ptr_eq {
  bool operator()(T const* lhs, T const* rhs) const {
    if (lhs == nullptr) {
      return rhs == nullptr;
    } else if (rhs == nullptr) {
      return lhs == nullptr;
    } else {
      return cista::equal_to<T>{}(*lhs, *rhs);
    }
  }
};

}  // namespace motis