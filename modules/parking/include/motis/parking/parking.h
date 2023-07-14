#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "motis/ppr/profile_info.h"

#include "motis/core/schedule/station_lookup.h"
#include "motis/module/module.h"

namespace motis::parking {

struct stations;

struct parking : public motis::module::module {
  parking();
  ~parking() override;

  parking(parking const&) = delete;
  parking& operator=(parking const&) = delete;

  parking(parking&&) = delete;
  parking& operator=(parking&&) = delete;

  void import(motis::module::import_dispatcher&) override;
  void init(motis::module::registry&) override;

  bool import_successful() const override { return import_successful_; }

private:
  std::filesystem::path module_data_dir() const;
  std::string db_file() const;

  // import
  int max_walk_duration_{10};
  std::size_t edge_rtree_max_size_{1024UL * 1024 * 1024 * 3};
  std::size_t area_rtree_max_size_{1024UL * 1024 * 1024};
  bool lock_rtrees_{false};
  bool import_osm_{true};
  bool ppr_exact_{true};

  std::size_t db_max_size_{static_cast<std::size_t>(1024) * 1024 * 1024 * 512};

  std::vector<std::string> parkendd_endpoints_;
  unsigned parkendd_update_interval_{300};  // seconds

  struct impl;
  std::unique_ptr<impl> impl_;
  bool import_successful_{false};
  std::map<std::string, ::motis::ppr::profile_info> ppr_profiles_;
  station_lookup const* stations_{nullptr};
};

}  // namespace motis::parking
