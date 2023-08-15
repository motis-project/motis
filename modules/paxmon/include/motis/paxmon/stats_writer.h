#pragma once

#include <fstream>
#include <string>

#include "motis/paxmon/csv_writer.h"
#include "motis/paxmon/statistics.h"

namespace motis::paxmon {

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

  file_csv_writer csv_;
};

}  // namespace motis::paxmon
