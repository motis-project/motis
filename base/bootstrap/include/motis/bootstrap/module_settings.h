#pragma once

#include <string>
#include <vector>

#include "boost/program_options.hpp"

#include "conf/configuration.h"

namespace motis::bootstrap {

class module_settings : public conf::configuration {
public:
  explicit module_settings(std::vector<std::string> modules = {})
      : conf::configuration("Module Settings"), modules_{std::move(modules)} {
    param(modules_, "modules", "List of modules to load");
    param(exclude_modules_, "exclude_modules", "List of modules to exclude");
  }

  bool is_module_active(std::string const& module) const {
    auto const& yes = modules_;
    auto const& no = exclude_modules_;
    return std::find(begin(yes), end(yes), module) != end(yes) &&
           std::find(begin(no), end(no), module) == end(no);
  }

  std::vector<std::string> modules_;
  std::vector<std::string> exclude_modules_;
};

}  // namespace motis::bootstrap
