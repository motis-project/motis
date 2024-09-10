#pragma once

#include "prometheus/counter.h"
#include "prometheus/family.h"
#include "prometheus/gauge.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"

namespace motis::nigiri {

struct metrics {
  explicit metrics(prometheus::Registry& registry);

  prometheus::Registry& registry_;

  prometheus::Family<prometheus::Counter>& request_counter_family_;
  prometheus::Counter& pretrip_requests_;
  prometheus::Counter& ontrip_station_requests_;

  prometheus::Histogram& via_count_;

  prometheus::Family<prometheus::Histogram>& routing_time_family_;
  prometheus::Histogram& pretrip_routing_time_;
  prometheus::Histogram& ontrip_station_routing_time_;

  prometheus::Histogram& pretrip_interval_extensions_;

  prometheus::Histogram& reconstruction_errors_;

  prometheus::Family<prometheus::Counter>& gtfsrt_updates_requested_;
  prometheus::Family<prometheus::Counter>& gtfsrt_updates_successful_;
  prometheus::Family<prometheus::Counter>& gtfsrt_updates_error_;

  prometheus::Family<prometheus::Counter>& gtfsrt_total_entities_;
  prometheus::Family<prometheus::Counter>& gtfsrt_total_entities_success_;
  prometheus::Family<prometheus::Counter>& gtfsrt_total_entities_fail_;
  prometheus::Family<prometheus::Counter>& gtfsrt_unsupported_deleted_;
  prometheus::Family<prometheus::Counter>& gtfsrt_unsupported_vehicle_;
  prometheus::Family<prometheus::Counter>& gtfsrt_unsupported_alert_;
  prometheus::Family<prometheus::Counter>& gtfsrt_unsupported_no_trip_id_;
  prometheus::Family<prometheus::Counter>& gtfsrt_no_trip_update_;
  prometheus::Family<prometheus::Counter>& gtfsrt_trip_update_without_trip_;
  prometheus::Family<prometheus::Counter>& gtfsrt_trip_resolve_error_;
  prometheus::Family<prometheus::Counter>&
      gtfsrt_unsupported_schedule_relationship_;
  prometheus::Family<prometheus::Gauge>& gtfsrt_feed_timestamp_;
  prometheus::Family<prometheus::Gauge>& gtfsrt_last_update_timestamp_;
};

}  // namespace motis::nigiri
