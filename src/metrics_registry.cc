#include "motis/metrics_registry.h"
#include "prometheus/histogram.h"

namespace motis {

metrics_registry::metrics_registry()
    : metrics_registry(
          prometheus::Histogram::BucketBoundaries{
              0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 75, 100, 1000},
          prometheus::Histogram::BucketBoundaries{
              0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60}) {}

metrics_registry::metrics_registry(
    prometheus::Histogram::BucketBoundaries event_boundaries,
    prometheus::Histogram::BucketBoundaries time_boundaries)
    : registry_{prometheus::Registry()},
      routing_requests_{prometheus::BuildCounter()
                            .Name("motis_routing_requests_total")
                            .Help("Number of routing requests")
                            .Register(registry_)
                            .Add({})},
      routing_journeys_found_{prometheus::BuildCounter()
                                  .Name("motis_routing_journeys_found_total")
                                  .Help("Number of journey results")
                                  .Register(registry_)
                                  .Add({})},
      routing_odm_journeys_found_{
          prometheus::BuildHistogram()
              .Name("motis_routing_odm_events_found")
              .Help("Number of journey results including an ODM part")
              .Register(registry_)},
      routing_odm_journeys_found_blacklist_{routing_odm_journeys_found_.Add(
          {{"stage", "blacklist"}},
          prometheus::Histogram::BucketBoundaries{0, 500, 1000, 2000, 3000,
                                                  4000, 5000, 10000, 15000,
                                                  20000, 50000})},
      routing_odm_journeys_found_whitelist_{routing_odm_journeys_found_.Add(
          {{"stage", "whitelist"}}, event_boundaries)},
      routing_odm_journeys_found_non_dominated_pareto_{
          routing_odm_journeys_found_.Add({{"stage", "non_dominated_pareto"}},
                                          event_boundaries)},
      routing_odm_journeys_found_non_dominated_cost_{
          routing_odm_journeys_found_.Add({{"stage", "non_dominated_cost"}},
                                          event_boundaries)},
      routing_odm_journeys_found_non_dominated_prod_{
          routing_odm_journeys_found_.Add({{"stage", "non_dominated_prod"}},
                                          event_boundaries)},
      routing_odm_journeys_found_non_dominated_{routing_odm_journeys_found_.Add(
          {{"stage", "non_dominated"}}, event_boundaries)},
      routing_journey_duration_seconds_{
          prometheus::BuildHistogram()
              .Name("motis_routing_journey_duration_seconds")
              .Help("Journey duration statistics")
              .Register(registry_)
              .Add({},
                   prometheus::Histogram::BucketBoundaries{
                       300, 600, 1200, 1800, 3600, 7200, 10800, 14400, 18000,
                       21600, 43200, 86400})},
      routing_execution_duration_seconds_{
          prometheus::BuildHistogram()
              .Name("motis_routing_execution_duration_seconds")
              .Help("Routing execution duration statistics")
              .Register(registry_)},
      routing_execution_duration_seconds_init_{
          routing_execution_duration_seconds_.Add({{"stage", "init"}},
                                                  time_boundaries)},
      routing_execution_duration_seconds_blacklisting_{
          routing_execution_duration_seconds_.Add({{"stage", "blacklisting"}},
                                                  time_boundaries)},
      routing_execution_duration_seconds_preparing_{
          routing_execution_duration_seconds_.Add({{"stage", "preparing"}},
                                                  time_boundaries)},
      routing_execution_duration_seconds_routing_{
          routing_execution_duration_seconds_.Add({{"stage", "routing"}},
                                                  time_boundaries)},
      routing_execution_duration_seconds_whitelisting_{
          routing_execution_duration_seconds_.Add({{"stage", "whitelisting"}},
                                                  time_boundaries)},
      routing_execution_duration_seconds_mixing_{
          routing_execution_duration_seconds_.Add({{"stage", "mixing"}},
                                                  time_boundaries)},
      routing_execution_duration_seconds_total_{
          routing_execution_duration_seconds_.Add({{"stage", "total"}},
                                                  time_boundaries)},
      current_trips_running_scheduled_count_{
          prometheus::BuildGauge()
              .Name("current_trips_running_scheduled_count")
              .Help("The number of currently running transports")
              .Register(registry_)},
      current_trips_running_scheduled_with_realtime_count_{
          prometheus::BuildGauge()
              .Name("current_trips_running_scheduled_with_realtime_count")
              .Help("The number of currently running transports that have RT "
                    "data")
              .Register(registry_)},
      total_trips_with_realtime_count_{
          prometheus::BuildGauge()
              .Name("total_trips_with_realtime_count")
              .Help("The total number of transports that have RT data")
              .Register(registry_)
              .Add({})},
      timetable_first_day_timestamp_{
          prometheus::BuildGauge()
              .Name("nigiri_timetable_first_day_timestamp_seconds")
              .Help("Day of the first transport in unixtime")
              .Register(registry_)},
      timetable_last_day_timestamp_{
          prometheus::BuildGauge()
              .Name("nigiri_timetable_last_day_timestamp_seconds")
              .Help("Day of the last transport in unixtime")
              .Register(registry_)},
      timetable_locations_count_{
          prometheus::BuildGauge()
              .Name("nigiri_timetable_locations_count")
              .Help("The number of locations in the timetable")
              .Register(registry_)},
      timetable_trips_count_{prometheus::BuildGauge()
                                 .Name("nigiri_timetable_trips_count")
                                 .Help("The number of trips in the timetable")
                                 .Register(registry_)},
      timetable_transports_x_days_count_{
          prometheus::BuildGauge()
              .Name("nigiri_timetable_transports_x_days_count")
              .Help("The number of transports x service days in the timetable")
              .Register(registry_)} {}

metrics_registry::~metrics_registry() = default;

}  // namespace motis
