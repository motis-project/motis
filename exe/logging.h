#include <cctype>
#include <algorithm>
#include <string>
#include <string_view>

#include "utl/logging.h"

#include "nigiri/logging.h"

namespace motis {

inline bool set_log_level(std::string& ll) {
  std::transform(ll.begin(), ll.end(), ll.begin(),
                 [](unsigned char const c) { return std::toupper(c); });
  if (ll == std::string_view{"ERROR"}) {
    utl::log_verbosity = utl::log_level::error;
    nigiri::s_verbosity = nigiri::log_lvl::error;
  } else if (ll == std::string_view{"INFO"}) {
    utl::log_verbosity = utl::log_level::info;
    nigiri::s_verbosity = nigiri::log_lvl::info;
  } else if (ll == std::string_view{"DEBUG"}) {
    utl::log_verbosity = utl::log_level::debug;
    nigiri::s_verbosity = nigiri::log_lvl::debug;
  } else {
    return false;
  }
  return true;
}

}  // namespace motis
