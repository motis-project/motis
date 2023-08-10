#pragma once

#include <cinttypes>
#include <memory>
#include <string>

#include "conf/date_time.h"
#include "conf/duration.h"

#include "motis/module/module.h"

#include "motis/ris/rabbitmq_config.h"

namespace motis::ris {

struct config {
  std::string db_path_{"ris.mdb"};
  std::vector<std::string> input_;
  conf::time init_time_{0};
  bool clear_db_ = false;
  conf::duration init_purge_{};
  size_t db_max_size_{static_cast<size_t>(1024) * 1024 * 1024 * 512};
  bool instant_forward_{false};
  bool gtfs_is_addition_skip_allowed_{true};
  unsigned gtfs_rt_update_interval_{60};
  std::string http_proxy_;
  rabbitmq_config ribasis_fahrt_;
  rabbitmq_config ribasis_formation_;
};

struct ris : public motis::module::module {
  ris();
  ~ris() override;

  ris(ris const&) = delete;
  ris& operator=(ris const&) = delete;

  ris(ris&&) = delete;
  ris& operator=(ris&&) = delete;

  void reg_subc(motis::module::subc_reg&) override;
  void init(motis::module::registry&) override;
  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override;

  void stop_io() override;

private:
  struct impl;
  std::unique_ptr<impl> impl_;
  config config_;

  bool import_successful_{false};
};

}  // namespace motis::ris
