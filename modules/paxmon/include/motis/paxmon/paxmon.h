#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "conf/date_time.h"

#include "motis/module/module.h"

#include "motis/paxmon/loader/loader_result.h"
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
  loader::loader_result load_journeys(std::string const& file);
  void load_capacity_files();
  motis::module::msg_ptr rt_update(motis::module::msg_ptr const& msg);
  void rt_updates_applied();
  motis::module::msg_ptr add_groups(motis::module::msg_ptr const& msg);
  motis::module::msg_ptr remove_groups(motis::module::msg_ptr const& msg);

  std::vector<std::string> journey_files_;
  std::vector<std::string> capacity_files_;
  std::string stats_file_{"paxmon_stats.csv"};
  std::string capacity_match_log_file_{};
  std::string journey_match_log_file_{};
  std::string initial_over_capacity_report_file_{};
  std::string initial_broken_report_file_{};
  std::string initial_reroute_query_file_{};
  std::string initial_reroute_router_{"/tripbased"};
  conf::holder<std::time_t> start_time_{};
  conf::holder<std::time_t> end_time_{};
  int time_step_{60};
  std::uint16_t match_tolerance_{0};
  bool reroute_unmatched_{false};
  int arrival_delay_threshold_{20};
  bool check_graph_times_{false};
  bool check_graph_integrity_{false};
  std::string mcfp_scenario_dir_{};
  int mcfp_scenario_min_broken_groups_{500};

  paxmon_data data_;
  system_statistics system_stats_;
  tick_statistics tick_stats_;
  std::unique_ptr<stats_writer> stats_writer_;
  bool write_mcfp_scenarios_{false};
};

}  // namespace motis::paxmon
