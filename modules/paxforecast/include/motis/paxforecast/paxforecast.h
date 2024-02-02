#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "conf/date_time.h"

#include "motis/core/schedule/time.h"
#include "motis/module/module.h"

#include "motis/paxforecast/routing_cache.h"
#include "motis/paxforecast/stats_writer.h"
#include "motis/paxforecast/universe_storage.h"

#include "motis/paxforecast/behavior/probabilistic/passenger_behavior.h"

namespace motis::paxforecast {

struct universe_data;

struct paxforecast : public motis::module::module {
  paxforecast();
  ~paxforecast() override;

  paxforecast(paxforecast const&) = delete;
  paxforecast& operator=(paxforecast const&) = delete;

  paxforecast(paxforecast&&) = delete;
  paxforecast& operator=(paxforecast&&) = delete;

  void init(motis::module::registry&) override;

  std::string forecast_filename_;
  std::ofstream forecast_file_;

  std::string behavior_stats_filename_;
  std::ofstream behavior_stats_file_;

  std::string routing_cache_filename_;
  routing_cache routing_cache_;

  duration min_delay_improvement_{5};
  bool revert_forecasts_{false};
  float probability_threshold_{0.01F};
  float uninformed_pax_{0.F};
  float major_delay_switch_{0.F};

  bool allow_start_metas_{false};
  bool allow_dest_metas_{false};

  std::string behavior_file_;
  std::string stats_file_;
  std::unique_ptr<stats_writer> stats_writer_;
  universe_storage<universe_data> universe_storage_;

  std::unique_ptr<behavior::probabilistic::passenger_behavior> behavior_;
};

}  // namespace motis::paxforecast
