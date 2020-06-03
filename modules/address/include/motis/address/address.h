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

  void import(motis::module::registry&) override;
  void init(motis::module::registry&) override;

  bool import_successful() const override { return import_successful_; }

private:
  std::string db_file() const;

  struct impl;
  std::unique_ptr<impl> impl_;
  bool import_successful_{false};
};

}  // namespace motis::address
