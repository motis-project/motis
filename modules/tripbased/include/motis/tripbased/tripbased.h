#pragma once

#include "motis/module/module.h"

namespace motis::tripbased {

struct tb_data;

struct tripbased : public motis::module::module {

  tripbased();
  ~tripbased() override;

  tripbased(tripbased const&) = delete;
  tripbased& operator=(tripbased const&) = delete;

  tripbased(tripbased&&) = delete;
  tripbased& operator=(tripbased&&) = delete;

  void import(motis::module::registry&) override;
  void init(motis::module::registry&) override;

  bool import_successful() const override;

  tb_data const* get_data() const;

private:
  bool use_data_file_{true};

  bool import_successful_{false};

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::tripbased
