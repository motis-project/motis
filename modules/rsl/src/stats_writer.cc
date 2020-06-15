#include "motis/rsl/stats_writer.h"

namespace motis::rsl {

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
       << "affected_groups"
       << "affected_passengers"
       << "ok_groups"
       << "broken_groups"
       << "broken_passengers"
       << "combined_groups"
       << "combined_groups_size"
       //
       << "total_ok_groups"
       << "total_broken_groups"
       //
       << "tracked_ok_groups"
       << "tracked_broken_groups" << end_row;
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
       << ts.affected_groups_ << ts.affected_passengers_ << ts.ok_groups_
       << ts.broken_groups_ << ts.broken_passengers_ << ts.combined_groups_
       << ts.combined_groups_size_
       //
       << ts.total_ok_groups_
       << ts.total_broken_groups_
       //
       << ts.tracked_ok_groups_ << ts.tracked_broken_groups_ << end_row;
}

void stats_writer::flush() { csv_.flush(); }

}  // namespace motis::rsl
