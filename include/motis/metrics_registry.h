#pragma once

#include "prometheus/counter.h"
#include "prometheus/family.h"
#include "prometheus/gauge.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"

namespace motis {

struct metrics_registry {
  metrics_registry();
  ~metrics_registry();
  prometheus::Registry registry_;
  prometheus::Counter& routing_requests_;
  prometheus::Counter& routing_journeys_found_;
  prometheus::Family<prometheus::Histogram>& routing_odm_journeys_found_;
  prometheus::Histogram& routing_odm_journeys_found_blacklist_;
  prometheus::Histogram& routing_odm_journeys_found_whitelist_;
  prometheus::Histogram& routing_odm_journeys_found_non_dominated_pareto_;
  prometheus::Histogram& routing_odm_journeys_found_non_dominated_cost_;
  prometheus::Histogram& routing_odm_journeys_found_non_dominated_prod_;
  prometheus::Histogram& routing_odm_journeys_found_non_dominated_;
  prometheus::Histogram& routing_journey_duration_seconds_;
  prometheus::Family<prometheus::Histogram>&
      routing_execution_duration_seconds_;
  prometheus::Histogram& routing_execution_duration_seconds_init_;
  prometheus::Histogram& routing_execution_duration_seconds_blacklisting_;
  prometheus::Histogram& routing_execution_duration_seconds_preparing_;
  prometheus::Histogram& routing_execution_duration_seconds_routing_;
  prometheus::Histogram& routing_execution_duration_seconds_whitelisting_;
  prometheus::Histogram& routing_execution_duration_seconds_mixing_;
  prometheus::Histogram& routing_execution_duration_seconds_total_;

private:
  metrics_registry(prometheus::Histogram::BucketBoundaries event_boundaries,
                   prometheus::Histogram::BucketBoundaries time_boundaries);
};

}  // namespace motis
