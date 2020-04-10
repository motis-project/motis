#pragma once

#include "motis/module/module.h"

namespace motis::parking {

struct parking : public motis::module::module {
  parking();
  ~parking() override;

  parking(parking const&) = delete;
  parking& operator=(parking const&) = delete;

  parking(parking&&) = delete;
  parking& operator=(parking&&) = delete;

  void init(motis::module::registry&) override;

private:
  std::string parking_file_{"parking.txt"};
  std::string footedges_db_file_{"parking_footedges.db"};
  std::size_t db_max_size_{static_cast<std::size_t>(1024) * 1024 * 1024 * 512};

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::parking
