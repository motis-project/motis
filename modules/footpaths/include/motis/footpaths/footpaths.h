#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

#include "motis/module/module.h"

namespace motis::footpaths {

struct footpaths : public motis::module::module {
  footpaths();
  ~footpaths() override;

  footpaths(footpaths const&) = delete;
  footpaths& operator=(footpaths const&) = delete;

  footpaths(footpaths&&) = delete;
  footpaths& operator=(footpaths&&) = delete;

  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override { return import_successful_; };

private:
  // directories
  std::filesystem::path module_data_dir() const;
  std::string db_file() const;

  // initialize ppr routing data
  static const std::size_t edge_rtree_max_size_{1024UL * 1024 * 1024 * 3};
  static const std::size_t area_rtree_max_size_{1024UL * 1024 * 1024};
  static const bool lock_rtree_{false};
  // bool ppr_exact_{false};

  std::size_t db_max_size_{static_cast<std::size_t>(1024) * 1024 * 1024 * 512};

  struct impl;
  std::unique_ptr<impl> impl_;
  bool import_successful_{false};
};

}  // namespace motis::footpaths
