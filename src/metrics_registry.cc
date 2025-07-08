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
              .Help("The number of currently running trips")
              .Register(registry_)},
      current_trips_running_scheduled_with_realtime_count_{
          prometheus::BuildGauge()
              .Name("current_trips_running_scheduled_with_realtime_count")
              .Help("The number of currently running trips that have RT data")
              .Register(registry_)} {}

metrics_registry::~metrics_registry() = default;

}  // namespace motis
