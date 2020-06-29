#pragma once

#include <memory>
#include <string>
#include <vector>

#include "conf/date_time.h"

#include "motis/module/module.h"

#include "motis/paxmon/paxmon_data.h"
#include "motis/paxmon/statistics.h"
#include "motis/paxmon/stats_writer.h"

namespace motis::paxmon {

struct paxmon : public motis::module::module {
  paxmon();
  ~paxmon() override;

  paxmon(paxmon const&) = delete;
  paxmon& operator=(paxmon const&) = delete;

  paxmon(paxmon&&) = delete;
  paxmon& operator=(paxmon&&) = delete;

  void init(motis::module::registry&) override;

private:
  void load_journeys();
  void load_capacity_file();
  motis::module::msg_ptr rt_update(motis::module::msg_ptr const& msg);
  void rt_updates_applied();

  std::string journey_file_{"paxmon_journeys.txt"};
  std::string capacity_file_{};
  std::string stats_file_{"paxmon_stats.csv"};
  conf::holder<std::time_t> start_time_{};
  conf::holder<std::time_t> end_time_{};

  paxmon_data data_;
  system_statistics system_stats_;
  tick_statistics tick_stats_;
  std::unique_ptr<stats_writer> stats_writer_;
};

}  // namespace motis::paxmon
