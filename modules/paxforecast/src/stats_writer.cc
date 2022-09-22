#include "motis/paxforecast/stats_writer.h"

using motis::paxmon::end_row;

namespace motis::paxforecast {

stats_writer::stats_writer(const std::string& filename) : csv_{filename} {
  write_header();
}

stats_writer::~stats_writer() { csv_.flush(); }

void stats_writer::write_header() {
  csv_ << "system_time"
       //
       << "monitoring_events"
       << "group_routes"
       << "combined_groups"
       << "major_delay_group_routes"
       //
       << "routing_requests"
       << "alternatives_found"
       //
       << "rerouted_group_routes"
       << "removed_group_routes"
       << "major_delay_group_routes_with_alternatives"
       //
       << "t_find_alternatives"
       << "t_add_alternatives"
       << "t_passenger_behavior"
       << "t_calc_load_forecast"
       << "t_load_forecast_fbs"
       << "t_write_load_forecast"
       << "t_publish_load_forecast"
       << "t_total_load_forecast"
       << "t_update_tracked_groups"
       << "t_total"
       //
       << end_row;
}

void stats_writer::write_tick(const tick_statistics& ts) {
  csv_ << ts.system_time_
       //
       << ts.monitoring_events_ << ts.group_routes_ << ts.combined_groups_
       << ts.major_delay_group_routes_
       //
       << ts.routing_requests_
       << ts.alternatives_found_
       //
       << ts.rerouted_group_routes_ << ts.removed_group_routes_
       << ts.major_delay_group_routes_with_alternatives_
       //
       << ts.t_find_alternatives_ << ts.t_add_alternatives_
       << ts.t_passenger_behavior_ << ts.t_calc_load_forecast_
       << ts.t_load_forecast_fbs_ << ts.t_write_load_forecast_
       << ts.t_publish_load_forecast_ << ts.t_total_load_forecast_
       << ts.t_update_tracked_groups_
       << ts.t_total_
       //
       << end_row;
}

void stats_writer::flush() { csv_.flush(); }

}  // namespace motis::paxforecast
