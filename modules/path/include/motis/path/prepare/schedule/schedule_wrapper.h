#pragma once

#include <map>
#include <string>
#include <vector>

#include "utl/parser/buffer.h"

#include "motis/path/prepare/schedule/station_sequences.h"

namespace motis::path {

struct schedule_wrapper {
  explicit schedule_wrapper(std::string const& schedule_path);

  mcd::vector<station_seq> load_station_sequences(
      std::string const& prefix) const;

  utl::buffer schedule_buffer_;
};

}  // namespace motis::path
