#pragma once

#include "motis/module/module.h"

namespace motis::address {

struct address : public motis::module::module {
  address();
  ~address() override;

  address(address const&) = delete;
  address& operator=(address const&) = delete;

  address(address&&) = delete;
  address& operator=(address&&) = delete;

  void init(motis::module::registry&) override;

private:
  std::string db_path_{"address_db"};

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::address
