#pragma once

#include <cstdio>

#ifdef _MSC_VER
#include <io.h>
#else
#include <unistd.h>
#endif

namespace motis::eval {

inline bool is_terminal(std::FILE* f) {
#ifdef _MSC_VER
  return _isatty(_fileno(f)) != 0;
#else
  return isatty(fileno(f)) != 0;
#endif
}

}  // namespace motis::eval
