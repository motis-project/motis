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

  void import(motis::module::registry& reg) override;
  void init(motis::module::registry&) override;

  bool import_successful() const override;

  bool use_coastline_{false};
  std::string profile_path_;

  struct data;
  std::unique_ptr<data> data_;
};

}  // namespace motis::tiles
