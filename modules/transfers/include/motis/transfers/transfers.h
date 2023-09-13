#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>

#include "motis/module/module.h"

namespace motis::transfers {

struct transfers : public motis::module::module {
  transfers();
  ~transfers() override;

  transfers(transfers const&) = delete;
  transfers& operator=(transfers const&) = delete;

  transfers(transfers&&) = delete;
  transfers& operator=(transfers&&) = delete;

  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override { return import_successful_; };

private:
  // directories
  std::filesystem::path module_data_dir() const;
  std::filesystem::path db_file() const;

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

}  // namespace motis::transfers
