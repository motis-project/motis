#pragma once

#include <string>
#include <vector>

#include "conf/configuration.h"

namespace motis::bootstrap {

struct import_settings : conf::configuration {
  explicit import_settings(std::vector<std::string> import_paths = {})
      : configuration("Import Options", "import"),
        import_paths_{std::move(import_paths)} {
    param(import_paths_, "paths",
          "input paths to process. expected format: tag-options:path "
          "(brackets are optional if options empty)");
    param(data_directory_, "data_dir", "directory for preprocessing output");
    param(require_successful_, "require_successful",
          "exit if import is not successful for all modules");
  }

  import_settings(import_settings const&) = delete;
  import_settings(import_settings&&) = delete;
  import_settings& operator=(import_settings const&) = delete;
  import_settings& operator=(import_settings&&) = delete;
  ~import_settings() override = default;

  std::vector<std::string> import_paths_;
  std::string data_directory_{"data"};
  bool require_successful_{true};
};

}  // namespace motis::bootstrap