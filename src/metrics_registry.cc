#include "motis/metrics_registry.h"
#include "prometheus/histogram.h"

namespace motis {

metrics_registry::metrics_registry()
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
          prometheus::BuildCounter()
              .Name("motis_routing_odm_journeys_found_total")
              .Help("Number of journey results including an ODM part")
              .Register(registry_)
              .Add({})},
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
              .Register(registry_)
              .Add({},
                   prometheus::Histogram::BucketBoundaries{
                       0.5, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 45, 60})} {}

metrics_registry::~metrics_registry() = default;

}  // namespace motis
