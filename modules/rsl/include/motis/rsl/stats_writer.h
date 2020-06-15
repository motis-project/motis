#pragma once

#include <fstream>
#include <string>

#include "motis/rsl/csv_writer.h"
#include "motis/rsl/statistics.h"

namespace motis::rsl {

struct stats_writer {
  explicit stats_writer(std::string const& filename);
  ~stats_writer();

  void write_tick(tick_statistics const& ts);
  void flush();

private:
  void write_header();

  csv_writer csv_;
};

}  // namespace motis::rsl
