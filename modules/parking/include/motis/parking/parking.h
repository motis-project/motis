#pragma once

#include "boost/filesystem.hpp"

#include "motis/module/module.h"

namespace motis::parking {

struct parking : public motis::module::module {
  parking();
  ~parking() override;

  parking(parking const&) = delete;
  parking& operator=(parking const&) = delete;

  parking(parking&&) = delete;
  parking& operator=(parking&&) = delete;

  void import(motis::module::registry& reg) override;
  void init(motis::module::registry&) override;

  bool import_successful() const override { return import_successful_; }

private:
  boost::filesystem::path module_data_dir() const;
  std::string parking_file() const;
  std::string footedges_db_file() const;
  std::string stations_per_parking_file() const;

  // import
  int max_walk_duration_{10};
  std::size_t edge_rtree_max_size_{1024UL * 1024 * 1024 * 3};
  std::size_t area_rtree_max_size_{1024UL * 1024 * 1024};
  bool lock_rtrees_{false};

  std::size_t db_max_size_{static_cast<std::size_t>(1024) * 1024 * 1024 * 512};

  struct impl;
  std::unique_ptr<impl> impl_;
  bool import_successful_{false};
};

}  // namespace motis::parking
