#pragma once

#include <cinttypes>
#include <string>

#include "conf/date_time.h"

#include "motis/module/module.h"

namespace motis::ris {

struct config {
  std::string gtfs_trip_ids_path_;
  std::string db_path_{"ris.mdb"};
  std::string input_{"ris"};
  conf::time init_time_{0};
  bool clear_db_ = false;
  size_t db_max_size_{static_cast<size_t>(1024) * 1024 * 1024 * 512};
  bool instant_forward_{false};
  bool gtfs_is_addition_skip_allowed_{true};
};

struct ris : public motis::module::module {
  ris();
  ~ris() override;

  ris(ris const&) = delete;
  ris& operator=(ris const&) = delete;

  ris(ris&&) = delete;
  ris& operator=(ris&&) = delete;

  void init(motis::module::registry&) override;

private:
  struct impl;
  std::unique_ptr<impl> impl_;
  config config_;
};

}  // namespace motis::ris
