#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "motis/core/schedule/schedule.h"

#include "motis/rsl/combined_passenger_group.h"
#include "motis/rsl/rsl_data.h"
#include "motis/rsl/simulation_result.h"

namespace motis::rsl::output {

struct log_output {
  explicit log_output(std::string const& filename);
  ~log_output();

  log_output(log_output const&) = delete;
  log_output& operator=(log_output const&) = delete;

  log_output(log_output&&) = delete;
  log_output& operator=(log_output&&) = delete;

  void write_broken_connection(
      schedule const& sched, rsl_data const& data,
      std::map<unsigned, std::vector<combined_passenger_group>> const&
          combined_groups,
      simulation_result const& sim_result);

  void flush();

  struct writer;
  std::unique_ptr<writer> writer_;
};

}  // namespace motis::rsl::output
