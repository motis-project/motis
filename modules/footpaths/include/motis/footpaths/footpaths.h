#pragma once

#include "motis/ppr/profile_info.h"

#include "motis/module/module.h"

namespace fs = std::filesystem;

namespace motis::footpaths {

struct footpaths : public motis::module::module {
  footpaths();
  ~footpaths() override;

  footpaths(footpaths const&) = delete;
  footpaths& operator=(footpaths const&) = delete;

  footpaths(footpaths&&) = delete;
  footpaths& operator=(footpaths&&) = delete;

  void init(motis::module::registry&) override;
  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override { return import_successful_; };

private:
  // directories
  fs::path module_data_dir() const;

  int max_walk_duration_{15};

  struct impl;
  std::unique_ptr<impl> impl_;
  std::map<std::string, motis::ppr::profile_info> ppr_profiles_;
  std::map<std::string, size_t> ppr_profile_pos_;
  bool import_successful_{false};
};

}  // namespace motis::footpaths
