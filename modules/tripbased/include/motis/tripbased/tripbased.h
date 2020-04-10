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

  void init(motis::module::registry&) override;

  tb_data const* get_data() const;

private:
  std::string data_file_{"trip_based.bin"};

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::tripbased
