#pragma once

#include "motis/footpaths/platforms.h"
#include "motis/ppr/profile_info.h"

#include "motis/module/module.h"

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
  std::filesystem::path module_data_dir() const;
  std::string db_file() const;

    // initialize ppr routing data
  std::size_t edge_rtree_max_size_{1024UL * 1024 * 1024 * 3};
  std::size_t area_rtree_max_size_{1024UL * 1024 * 1024};
  bool lock_rtrees_{false};
  // bool ppr_exact_{false};

  std::size_t db_max_size_{static_cast<std::size_t>(1024) * 1024 * 1024 * 512};

  struct impl;
  std::unique_ptr<impl> impl_;
  std::map<std::string, ppr::profile_info> ppr_profiles_;
  std::vector<ppr::profile_info> profiles_;
  std::map<std::string, size_t> ppr_profile_pos_;
  bool import_successful_{false};
};

}  // namespace motis::footpaths
