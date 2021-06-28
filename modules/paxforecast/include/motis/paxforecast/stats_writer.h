#pragma once

#include <fstream>
#include <string>

#include "motis/paxforecast/statistics.h"
#include "motis/paxmon/csv_writer.h"

namespace motis::paxforecast {

struct stats_writer {
  explicit stats_writer(std::string const& filename);
  ~stats_writer();

  stats_writer(stats_writer const&) = delete;
  stats_writer& operator=(stats_writer const&) = delete;
  stats_writer(stats_writer&&) = delete;
  stats_writer& operator=(stats_writer&&) = delete;

  void write_tick(tick_statistics const& ts);
  void flush();

private:
  void write_header();

  motis::paxmon::csv_writer csv_;
};

}  // namespace motis::paxforecast
