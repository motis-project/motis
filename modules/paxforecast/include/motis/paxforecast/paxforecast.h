#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "conf/date_time.h"

#include "motis/module/module.h"

#include "motis/paxforecast/routing_cache.h"
#include "motis/paxforecast/stats_writer.h"

namespace motis::paxforecast {

struct paxforecast : public motis::module::module {
  paxforecast();
  ~paxforecast() override;

  paxforecast(paxforecast const&) = delete;
  paxforecast& operator=(paxforecast const&) = delete;

  paxforecast(paxforecast&&) = delete;
  paxforecast& operator=(paxforecast&&) = delete;

  void init(motis::module::registry&) override;

private:
  void on_monitoring_event(motis::module::msg_ptr const& msg);

  std::string forecast_filename_;
  std::ofstream forecast_file_;

  std::string behavior_stats_filename_;
  std::ofstream behavior_stats_file_;

  std::string routing_cache_filename_;
  routing_cache routing_cache_;

  bool calc_load_forecast_{true};
  bool publish_load_forecast_{false};

  bool deterministic_mode_{false};

  std::string stats_file_{"paxforecast_stats.csv"};
  std::unique_ptr<stats_writer> stats_writer_;
};

}  // namespace motis::paxforecast
