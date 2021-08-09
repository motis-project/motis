#pragma once

#include "motis/module/module.h"

namespace motis::tiles {

struct tiles : public motis::module::module {
  tiles();
  ~tiles() override;

  tiles(tiles const&) = delete;
  tiles& operator=(tiles const&) = delete;

  tiles(tiles&&) = delete;
  tiles& operator=(tiles&&) = delete;

  void import(motis::module::import_dispatcher&) override;
  void init(motis::module::registry&) override;

  bool import_successful() const override;

  bool use_coastline_{false};
  std::string profile_path_;
  size_t db_size_{sizeof(void*) >= 8 ? 1024ULL * 1024 * 1024 * 1024
                                     : 256 * 1024 * 1024};
  size_t flush_threshold_{sizeof(void*) >= 8 ? 10'000'000 : 100'000};

  struct data;
  std::unique_ptr<data> data_;
};

}  // namespace motis::tiles
