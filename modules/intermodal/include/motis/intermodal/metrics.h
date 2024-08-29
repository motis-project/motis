#pragma once

#include "prometheus/counter.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"

namespace motis::intermodal {

struct metrics {
  prometheus::Registry& registry_;

  prometheus::Counter& fwd_requests_;
  prometheus::Counter& bwd_requests_;

  prometheus::Counter& foot_modes_;
  prometheus::Counter& foot_ppr_modes_;
  prometheus::Counter& bike_modes_;
  prometheus::Counter& car_modes_;
  prometheus::Counter& car_parking_modes_;
  prometheus::Counter& gbfs_modes_;

  prometheus::Histogram& mumo_edges_time_;
  prometheus::Histogram& total_time_;
};

}  // namespace motis::intermodal
