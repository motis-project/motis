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
#include "motis/paxmon/settings/journey_input_settings.h"
#include "motis/paxmon/statistics.h"
#include "motis/paxmon/stats_writer.h"
#include "motis/paxmon/universe.h"

namespace motis::paxmon {

struct paxmon : public motis::module::module {
  paxmon();
  ~paxmon() override;

  paxmon(paxmon const&) = delete;
  paxmon& operator=(paxmon const&) = delete;

  paxmon(paxmon&&) = delete;
  paxmon& operator=(paxmon&&) = delete;

  void reg_subc(motis::module::subc_reg&) override;
  void import(motis::module::import_dispatcher& reg) override;
  void init(motis::module::registry&) override;

  bool import_successful() const override { return import_successful_; }

private:
  void load_journeys();
  loader::loader_result load_journeys(std::string const& file);
  void load_capacity_files();
  motis::module::msg_ptr rt_update(motis::module::msg_ptr const& msg);
  void rt_updates_applied(motis::module::msg_ptr const& msg);
  void rt_updates_applied(universe& uv, schedule const& sched);
  void universe_gc() const;

  universe& primary_universe();

  std::vector<std::string> journey_files_;
  settings::journey_input_settings journey_input_settings_{};
  std::vector<std::string> capacity_files_;
  std::string generated_capacity_file_;
  std::string stats_file_;
  std::string capacity_match_log_file_{};
  std::string initial_over_capacity_report_file_{};
  std::string initial_broken_report_file_{};
  std::string initial_reroute_query_file_{};
  std::string initial_reroute_router_{"/tripbased"};
  conf::time start_time_{};
  conf::time end_time_{};
  int time_step_{60};
  bool reroute_unmatched_{false};
  int arrival_delay_threshold_{20};
  int preparation_time_{15};
  bool check_graph_times_{false};
  bool check_graph_integrity_{false};
  std::string mcfp_scenario_dir_{};
  int mcfp_scenario_min_broken_groups_{500};
  bool mcfp_scenario_include_trip_info_{false};
  bool graph_log_enabled_{false};

  paxmon_data data_;
  std::unique_ptr<stats_writer> stats_writer_;
  bool write_mcfp_scenarios_{false};
  bool import_successful_{true};
  bool initial_forward_done_{false};
};

}  // namespace motis::paxmon
