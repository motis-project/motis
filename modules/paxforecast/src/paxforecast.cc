#include "motis/paxforecast/paxforecast.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <utility>

#include "fmt/format.h"

#include "utl/verify.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/access/service_access.h"
#include "motis/module/context/get_schedule.h"
#include "motis/module/context/motis_call.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/context/motis_spawn.h"
#include "motis/module/message.h"

#include "motis/paxmon/data_key.h"
#include "motis/paxmon/messages.h"
#include "motis/paxmon/monitoring_event.h"
#include "motis/paxmon/paxmon_data.h"

#include "motis/paxforecast/alternatives.h"
#include "motis/paxforecast/behavior/behaviors.h"
#include "motis/paxforecast/combined_passenger_group.h"
#include "motis/paxforecast/measures/measures.h"
#include "motis/paxforecast/messages.h"
#include "motis/paxforecast/output/output.h"
#include "motis/paxforecast/simulate_behavior.h"

using namespace motis::module;
using namespace motis::routing;
using namespace motis::logging;
using namespace motis::rt;
using namespace motis::paxmon;

namespace motis::paxforecast {

paxforecast::paxforecast() : module("Passenger Forecast", "paxforecast") {
  param(log_file_, "log", "output file");
}

paxforecast::~paxforecast() = default;

void paxforecast::init(motis::module::registry& reg) {
  LOG(info) << "passenger forecast module loaded";

  log_output_ = std::make_unique<output::log_output>(log_file_);

  reg.subscribe("/paxmon/monitoring_update", [&](msg_ptr const& msg) {
    on_monitoring_event(msg);
    return nullptr;
  });
}

void paxforecast::on_monitoring_event(msg_ptr const& msg) {
  auto const& sched = get_schedule();
  auto& data = *get_shared_data<paxmon_data*>(motis::paxmon::DATA_KEY);

  auto const mon_update = motis_content(MonitoringUpdate, msg);

  auto const current_time =
      unix_to_motistime(sched.schedule_begin_, sched.system_time_);
  utl::verify(current_time != INVALID_TIME, "invalid current system time");

  std::map<unsigned, std::vector<combined_passenger_group>> combined_groups;

  for (auto const& event : *mon_update->events()) {
    if (event->type() == MonitoringEventType_NO_PROBLEM) {
      continue;
    }
    auto const& pg = data.get_passenger_group(event->group()->id());
    auto const localization =
        from_fbs(sched, event->localization_type(), event->localization());
    auto const destination_station_id =
        pg.compact_planned_journey_.destination_station_id();

    auto& destination_groups = combined_groups[destination_station_id];
    auto cpg = std::find_if(
        begin(destination_groups), end(destination_groups),
        [&](auto const& g) { return g.localization_ == localization; });
    if (cpg == end(destination_groups)) {
      destination_groups.emplace_back(combined_passenger_group{
          destination_station_id, pg.passengers_, localization, {&pg}, {}});
    } else {
      cpg->passengers_ += pg.passengers_;
      cpg->groups_.push_back(&pg);
    }
  }

  if (combined_groups.empty()) {
    return;
  }

  auto routing_requests = 0ULL;
  auto alternatives_found = 0ULL;

  {
    scoped_timer alt_timer{"find alternatives"};
    std::vector<ctx::future_ptr<ctx_data, void>> futures;
    for (auto& cgs : combined_groups) {
      auto const destination_station_id = cgs.first;
      for (auto& cpg : cgs.second) {
        ++routing_requests;
        futures.emplace_back(
            spawn_job_void([&sched, destination_station_id, &cpg] {
              cpg.alternatives_ = find_alternatives(
                  sched, destination_station_id, cpg.localization_);
            }));
      }
    }
    ctx::await_all(futures);
  }

  {
    scoped_timer alt_trips_timer{"add alternatives to graph"};
    for (auto& cgs : combined_groups) {
      for (auto& cpg : cgs.second) {
        alternatives_found += cpg.alternatives_.size();
        for (auto const& alt : cpg.alternatives_) {
          for (auto const& leg : alt.compact_journey_.legs_) {
            get_or_add_trip(sched, data, leg.trip_);
          }
        }
      }
    }
  }

  LOG(info) << "alternatives: " << routing_requests << " routing requests => "
            << alternatives_found << " alternatives";

  {
    scoped_timer sim_behavior_timer{"simulate behavior"};
    auto pb = behavior::passenger_behavior(
        behavior::score::weighted{1.0, 0.0},
        behavior::influence::fixed_acceptance{0.75},
        behavior::distribution::best_only{}, behavior::post::round{});
    auto announcements = std::vector<measures::please_use>{};
    auto sim_result = simulation_result{};
    auto cpg_allocations = std::vector<
        std::pair<combined_passenger_group*, std::vector<std::uint16_t>>>{};
    for (auto& cgs : combined_groups) {
      for (auto& cpg : cgs.second) {
        cpg_allocations.emplace_back(
            &cpg,
            simulate_behavior(sched, data, cpg, announcements, pb, sim_result));
      }
    }

    if (sim_result.is_over_capacity()) {
      auto const trips_over_capacity =
          sim_result.trips_over_capacity_with_edges();
      std::cout << "problem detected: " << sim_result.edge_count_over_capacity()
                << " edges over capacity ("
                << sim_result.total_passengers_over_capacity()
                << " passengers), " << trips_over_capacity.size()
                << " trips over capacity\n";
      for (auto const& [trp, edges] : trips_over_capacity) {
        auto const ci = trp->edges_->front()
                            ->m_.route_edge_.conns_.at(trp->lcon_idx_)
                            .full_con_->con_info_;
        std::cout << "trip over capacity: " << get_service_name(sched, ci)
                  << " (tn=" << ci->train_nr_ << "):";
        for (auto const e : edges) {
          std::cout << " " << e->passengers() << "[+"
                    << sim_result.additional_passengers_[e] << "]/"
                    << e->capacity();
        }
        std::cout << "\n";
      }
    }

    log_output_->write_broken_connection(sched, data, combined_groups,
                                         sim_result);
    log_output_->flush();

    auto const forecast_msg =
        make_passenger_forecast_msg(sched, data, cpg_allocations, sim_result);

    revert_simulated_behavior(sim_result);

    ctx::await_all(motis_publish(forecast_msg));
  }
}

}  // namespace motis::paxforecast
