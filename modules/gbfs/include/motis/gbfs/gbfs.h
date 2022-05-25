#pragma once

#include <memory>
#include <string>
#include <vector>

#include "motis/module/module.h"

namespace motis::gbfs {

struct config {
  unsigned update_interval_minutes_{5U};
  std::vector<std::string> urls_;
  size_t db_size_{sizeof(void*) >= 8 ? 1024ULL * 1024 * 1024 * 1024
                                     : 256 * 1024 * 1024};
};

struct gbfs : public motis::module::module {
  gbfs();
  ~gbfs() override;

  gbfs(gbfs const&) = delete;
  gbfs& operator=(gbfs const&) = delete;

  gbfs(gbfs&&) = delete;
  gbfs& operator=(gbfs&&) = delete;

  void init(motis::module::registry&) override;
  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override { return import_successful_; }

private:
  bool import_successful_{false};

  struct impl;
  std::unique_ptr<impl> impl_;
  config config_;
};

}  // namespace motis::gbfs
