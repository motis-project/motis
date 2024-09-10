#include "motis/nigiri/metrics.h"

namespace motis::nigiri {

auto const routing_time_buckets = prometheus::Histogram::BucketBoundaries{
    .05, .1,  .25, .5,  .75,  1.0,  2.0,  3.0,  4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 30.0, 45.0, 60.0};

metrics::metrics(prometheus::Registry& registry)
    : registry_{registry},
      request_counter_family_{prometheus::BuildCounter()
                                  .Name("nigiri_routing_requests_total")
                                  .Help("Number of routing requests")
                                  .Register(registry)},
      pretrip_requests_{request_counter_family_.Add({{"type", "pretrip"}})},
      ontrip_station_requests_{
          request_counter_family_.Add({{"type", "ontrip_station"}})},
      via_count_{
          prometheus::BuildHistogram()
              .Name("nigiri_via_count")
              .Help("Number of via stops per routing request")
              .Register(registry)
              .Add({}, prometheus::Histogram::BucketBoundaries{0, 1, 2})},
      routing_time_family_{prometheus::BuildHistogram()
                               .Name("nigiri_routing_time_seconds")
                               .Help("Total time per routing request")
                               .Register(registry)},
      pretrip_routing_time_{routing_time_family_.Add({{"type", "pretrip"}},
                                                     routing_time_buckets)},
      ontrip_station_routing_time_{routing_time_family_.Add(
          {{"type", "ontrip_station"}}, routing_time_buckets)},
      pretrip_interval_extensions_{
          prometheus::BuildHistogram()
              .Name("nigiri_interval_extensions")
              .Help("Number of interval extensions per routing request")
              .Register(registry)
              .Add({{"type", "pretrip"}},
                   prometheus::Histogram::BucketBoundaries{0, 1, 2, 3, 4, 5, 6,
                                                           7, 8, 9, 10})},
      reconstruction_errors_{
          prometheus::BuildHistogram()
              .Name("nigiri_reconstruction_errors")
              .Help("Number of journey reconstruction errors per routing "
                    "request")
              .Register(registry)
              .Add({},
                   prometheus::Histogram::BucketBoundaries{0, 1, 2, 3, 4, 5, 6,
                                                           7, 8, 9, 10})},
      gtfsrt_updates_requested_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_updates_requested_total")
              .Help("Number of update attempts of the GTFS-RT feed")
              .Register(registry)},
      gtfsrt_updates_successful_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_updates_successful_total")
              .Help("Number of successful updates of the GTFS-RT feed")
              .Register(registry)},
      gtfsrt_updates_error_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_updates_error_total")
              .Help("Number of failed updates of the GTFS-RT feed")
              .Register(registry)},
      gtfsrt_total_entities_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_total_entities_total")
              .Help("Total number of entities in the GTFS-RT feed")
              .Register(registry)},
      gtfsrt_total_entities_success_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_total_entities_success_total")
              .Help("Number of entities in the GTFS-RT feed that were "
                    "successfully processed")
              .Register(registry)},
      gtfsrt_total_entities_fail_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_total_entities_fail_total")
              .Help("Number of entities in the GTFS-RT feed that could not "
                    "be processed")
              .Register(registry)},
      gtfsrt_unsupported_deleted_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_unsupported_deleted_total")
              .Help("Number of unsupported deleted entities in the GTFS-RT "
                    "feed")
              .Register(registry)},
      gtfsrt_unsupported_vehicle_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_unsupported_vehicle_total")
              .Help("Number of unsupported vehicle entities in the GTFS-RT "
                    "feed")
              .Register(registry)},
      gtfsrt_unsupported_alert_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_unsupported_alert_total")
              .Help("Number of unsupported alert entities in the GTFS-RT feed")
              .Register(registry)},
      gtfsrt_unsupported_no_trip_id_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_unsupported_no_trip_id_total")
              .Help("Number of unsupported trips without trip id in the "
                    "GTFS-RT feed")
              .Register(registry)},
      gtfsrt_no_trip_update_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_no_trip_update_total")
              .Help("Number of unsupported trips without trip update in the "
                    "GTFS-RT feed")
              .Register(registry)},
      gtfsrt_trip_update_without_trip_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_trip_update_without_trip_total")
              .Help("Number of unsupported trip updates without trip in the "
                    "GTFS-RT feed")
              .Register(registry)},
      gtfsrt_trip_resolve_error_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_trip_resolve_error_total")
              .Help("Number of unresolved trips in the GTFS-RT feed")
              .Register(registry)},
      gtfsrt_unsupported_schedule_relationship_{
          prometheus::BuildCounter()
              .Name("nigiri_gtfsrt_unsupported_schedule_relationship_total")
              .Help("Number of unsupported schedule relationships in the "
                    "GTFS-RT feed")
              .Register(registry)},
      gtfsrt_feed_timestamp_{prometheus::BuildGauge()
                                 .Name("nigiri_gtfsrt_feed_timestamp_seconds")
                                 .Help("Timestamp of the GTFS-RT feed")
                                 .Register(registry)},
      gtfsrt_last_update_timestamp_{
          prometheus::BuildGauge()
              .Name("nigiri_gtfsrt_last_update_timestamp_seconds")
              .Help("Last update timestamp of the GTFS-RT feed")
              .Register(registry)} {}

}  // namespace motis::nigiri
