#pragma once

#include <memory>
#include <string>
#include <vector>

#include "conf/date_time.h"

#include "motis/module/module.h"

#include "motis/rsl/output/output.h"
#include "motis/rsl/rsl_data.h"
#include "motis/rsl/statistics.h"
#include "motis/rsl/stats_writer.h"

namespace motis::rsl {

struct rsl : public motis::module::module {
  rsl();
  ~rsl() override;

  rsl(rsl const&) = delete;
  rsl& operator=(rsl const&) = delete;

  rsl(rsl&&) = delete;
  rsl& operator=(rsl&&) = delete;

  void init(motis::module::registry&) override;

private:
  void load_journeys();
  void load_capacity_file();
  motis::module::msg_ptr rt_update(motis::module::msg_ptr const& msg);
  void rt_updates_applied();

  std::string journey_file_{"rsl_journeys.txt"};
  std::string capacity_file_{};
  std::string log_file_{"rsl_log.jsonl"};
  std::string stats_file_{"rsl_stats.csv"};
  conf::holder<std::time_t> start_time_{};
  conf::holder<std::time_t> end_time_{};

  rsl_data data_;
  std::unique_ptr<output::log_output> log_output_;
  system_statistics system_stats_;
  tick_statistics tick_stats_;
  std::unique_ptr<stats_writer> stats_writer_;
};

}  // namespace motis::rsl
