#pragma once

#include <string>
#include <vector>

#include "boost/program_options.hpp"

#include "conf/configuration.h"

namespace motis::bootstrap {

class module_settings : public conf::configuration {
public:
  explicit module_settings(std::vector<std::string> modules)
      : conf::configuration("Module Settings"), modules_{std::move(modules)} {
    param(modules_, "modules", "List of modules to load");
    param(exclude_modules_, "exclude_modules", "List of modules to exclude");
  }
  std::vector<std::string> modules_;
  std::vector<std::string> exclude_modules_;
};

}  // namespace motis::bootstrap
