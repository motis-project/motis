#pragma once

#include <string>
#include <vector>

#include "conf/configuration.h"

namespace motis::bootstrap {

struct import_settings : conf::configuration {
  explicit import_settings(std::vector<std::string> const& import_paths = {})
      : configuration("Import Options", "import"), import_paths_{import_paths} {
    param(import_paths_, "paths", "input paths to process");
    param(data_directory_, "data_dir", "directory for preprocessing output");
  }

  import_settings(import_settings const&) = delete;
  import_settings(import_settings&&) = delete;
  import_settings& operator=(import_settings const&) = delete;
  import_settings& operator=(import_settings&&) = delete;
  ~import_settings() override = default;

  std::vector<std::string> import_paths_;
  std::string data_directory_{"data"};
};

}  // namespace motis::bootstrap