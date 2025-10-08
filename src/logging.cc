#include "motis/logging.h"

#include <algorithm>
#include <iostream>
#include <string_view>

#include "fmt/ostream.h"

#include "utl/logging.h"

#include "nigiri/logging.h"

namespace motis {

int set_log_level(std::string_view log_lvl) {
  if (log_lvl == "error") {
    utl::log_verbosity = utl::log_level::error;
    nigiri::s_verbosity = nigiri::log_lvl::error;
  } else if (log_lvl == "info") {
    utl::log_verbosity = utl::log_level::info;
    nigiri::s_verbosity = nigiri::log_lvl::info;
  } else if (log_lvl == "debug") {
    utl::log_verbosity = utl::log_level::debug;
    nigiri::s_verbosity = nigiri::log_lvl::debug;
  } else {
    fmt::println(std::cerr, "Unsupported log level '{}'\n", log_lvl);
    return 1;
  }
  return 0;
};

int set_log_level(config const& c) {
  if (c.logging_ && c.logging_->log_level_) {
    return set_log_level(*c.logging_->log_level_);
  }
  return 0;
}

int set_log_level(std::string&& log_lvl) {
  // Support uppercase for command line option
  std::transform(log_lvl.begin(), log_lvl.end(), log_lvl.begin(),
                 [](unsigned char const c) { return std::tolower(c); });
  return set_log_level(log_lvl);
}

}  // namespace motis
