#include "motis/paxmon/stats_writer.h"

namespace motis::paxmon {

stats_writer::stats_writer(const std::string& filename) : csv_{filename} {
  write_header();
}

stats_writer::~stats_writer() { csv_.flush(); }

void stats_writer::write_header() {
  csv_ << "system_time"
       //
       << "rt_updates"
       << "rt_delay_updates"
       << "rt_reroute_updates"
       << "rt_track_updates"
       << "rt_free_text_updates"
       //
       << "rt_delay_event_updates"
       << "rt_delay_is_updates"
       << "rt_delay_propagation_updates"
       << "rt_delay_forecast_updates"
       << "rt_delay_repair_updates"
       << "rt_delay_schedule_updates"
       //
       << "affected_group_routes"
       << "ok_group_routes"
       << "broken_group_routes"
       << "major_delay_group_routes"
       //
       << "t_reachability"
       << "t_localization"
       << "t_update_load"
       << "t_fbs_events"
       << "t_publish"
       << "t_rt_updates_applied_total"
       //
       << end_row;
}

void stats_writer::write_tick(const tick_statistics& ts) {
  csv_ << ts.system_time_
       //
       << ts.rt_updates_ << ts.rt_delay_updates_ << ts.rt_reroute_updates_
       << ts.rt_track_updates_
       << ts.rt_free_text_updates_
       //
       << ts.rt_delay_event_updates_ << ts.rt_delay_is_updates_
       << ts.rt_delay_propagation_updates_ << ts.rt_delay_forecast_updates_
       << ts.rt_delay_repair_updates_
       << ts.rt_delay_schedule_updates_
       //
       << ts.affected_group_routes_ << ts.ok_group_routes_
       << ts.broken_group_routes_
       << ts.major_delay_group_routes_
       //
       << ts.t_reachability_ << ts.t_localization_ << ts.t_update_load_
       << ts.t_fbs_events_ << ts.t_publish_
       << ts.t_rt_updates_applied_total_
       //
       << end_row;
}

void stats_writer::flush() { csv_.flush(); }

}  // namespace motis::paxmon
