#pragma once

#include <string>
#include <vector>

#include "conf/configuration.h"

namespace motis::bootstrap {

struct remote_settings : public conf::configuration {
  remote_settings() : configuration("Remote Settings") {
    param(remotes_, "remotes", "List of remotes to connect to");
  }

  std::vector<std::pair<std::string, std::string>> get_remotes() const;

  std::vector<std::string> remotes_;
};

}  // namespace motis::bootstrap
