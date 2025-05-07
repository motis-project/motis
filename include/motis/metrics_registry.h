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
  prometheus::Counter& routing_odm_journeys_found_;
  prometheus::Histogram& routing_journey_duration_seconds_;
  prometheus::Histogram& routing_execution_duration_seconds_;
};

}  // namespace motis
