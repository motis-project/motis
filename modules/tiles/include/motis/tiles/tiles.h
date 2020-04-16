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

  void init(motis::module::registry&) override;

  std::string database_path_;

  struct data;
  std::unique_ptr<data> data_;
};

}  // namespace motis::tiles
