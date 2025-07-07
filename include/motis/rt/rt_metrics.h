#pragma once

#include "prometheus/counter.h"
#include "prometheus/family.h"
#include "prometheus/gauge.h"

#include "motis/metrics_registry.h"

namespace motis {

struct rt_metric_families {
  explicit rt_metric_families(prometheus::Registry& registry)
      : gtfsrt_updates_requested_{prometheus::BuildCounter()
                                      .Name("nigiri_gtfsrt_updates_requested_"
                                            "total")
                                      .Help("Number of update attempts of the "
                                            "GTFS-RT feed")
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
                .Help(
                    "Number of unsupported alert entities in the GTFS-RT feed")
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
                .Register(registry)},
        vdvaus_updates_requested_{prometheus::BuildCounter()
                                      .Name("nigiri_vdvaus_updates_requested_"
                                            "total")
                                      .Help("Number of update attempts of the "
                                            "VDV AUS feed")
                                      .Register(registry)},
        vdvaus_updates_successful_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_updates_successful_total")
                .Help("Number of successful updates of the VDV AUS feed")
                .Register(registry)},
        vdvaus_updates_error_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_updates_error_total")
                .Help("Number of failed updates of the VDV AUS feed")
                .Register(registry)},
        vdvaus_unsupported_additional_runs_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_unsupported_additional_runs_total")
                .Help("Number of unsupported additional runs in the VDV AUS "
                      "feed")
                .Register(registry)},
        vdvaus_unsupported_additional_stops_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_unsupported_additional_runs_total")
                .Help("Number of additional stops in the VDV AUS feed")
                .Register(registry)},
        vdvaus_current_matches_total_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_current_matches_total")
                .Help("Current number of unique run IDs for which matching "
                      "was performed")
                .Register(registry)},
        vdvaus_current_matches_non_empty_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_current_matches_non_empty_total")
                .Help("Current number of unique run IDs for which a matching "
                      "was performed and a non-empty result was achieved")
                .Register(registry)},
        vdvaus_total_runs_{prometheus::BuildGauge()
                               .Name("nigiri_vdvaus_total_runs_total")
                               .Help("Total number of runs in the VDV AUS feed")
                               .Register(registry)},
        vdvaus_complete_runs_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_complete_runs_total")
                .Help("Total number of complete runs in the VDV AUS feed")
                .Register(registry)},
        vdvaus_unique_runs_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_unique_runs_total")
                .Help("Total number of unique runs in the VDV AUS feed")
                .Register(registry)},
        vdvaus_matched_runs_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_matched_runs_total")
                .Help(
                    "Number of runs of the VDV AUS feed that could be matched "
                    "to transports in the timetable, i.e., found or looked "
                    "up by established mapping")
                .Register(registry)},
        vdvaus_multiple_matches_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_multiple_matches_total")
                .Help("Number of times a run of the VDV AUS feed could not be "
                      "matched to a transport in the timetable since there "
                      "were multiple transports with the same score")
                .Register(registry)},
        vdvaus_incomplete_not_seen_before_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_incomplete_not_seen_before_total")
                .Help(
                    "Number of times an incomplete run was encountered before "
                    "seeing a complete version of it in the VDV AUS feed")
                .Register(registry)},
        vdvaus_no_transport_found_at_stop_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_no_transport_found_at_stop_total")
                .Help("Number of times that no transport could be found at the "
                      "stop specified in the VDV AUS feed")
                .Register(registry)},
        vdvaus_total_stops_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_total_stops_total")
                .Help("Total number of stops in the VDV AUS feed")
                .Register(registry)},
        vdvaus_resolved_stops_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_resolved_stops_total")
                .Help("Number of stops that could be resolved to locations in "
                      "the timetable")
                .Register(registry)},
        vdvaus_runs_without_stops_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_runs_without_stops_total")
                .Help("Number of times a run without any stops was encountered "
                      "in the VDV AUS feed")
                .Register(registry)},
        vdvaus_cancelled_runs_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_cancelled_runs_total")
                .Help("Number of cancelled runs in the VDV AUS feed")
                .Register(registry)},
        vdvaus_skipped_vdv_stops_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_skipped_vdv_stops_total")
                .Help("Number of stops in the VDV AUS feed that had to be "
                      "skipped while updating a run since they had no "
                      "counterpart in the run of the timetable")
                .Register(registry)},
        vdvaus_excess_vdv_stops_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_excess_vdv_stops_total")
                .Help(
                    "Number of additional stops at the end of runs in VDV AUS "
                    "feed that had no corresponding stop in the run of the "
                    "timetable that was updated")
                .Register(registry)},
        vdvaus_updated_events_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_updated_events_total")
                .Help("Number of arrival/departure times "
                      "that were updated by the VDV AUS feed")
                .Register(registry)},
        vdvaus_propagated_delays_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_propagated_delays_total")
                .Help("Number of delay propagations by the VDV AUS feed")
                .Register(registry)},
        vdvaus_feed_timestamp_{prometheus::BuildGauge()
                                   .Name("nigiri_vdvaus_feed_timestamp_seconds")
                                   .Help("Timestamp of the VDV AUS feed")
                                   .Register(registry)},
        vdvaus_last_update_timestamp_{
            prometheus::BuildGauge()
                .Name("nigiri_vdvaus_last_update_timestamp_seconds")
                .Help("Last update timestamp of the VDV AUS feed")
                .Register(registry)} {}

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

  prometheus::Family<prometheus::Counter>& vdvaus_updates_requested_;
  prometheus::Family<prometheus::Counter>& vdvaus_updates_successful_;
  prometheus::Family<prometheus::Counter>& vdvaus_updates_error_;

  prometheus::Family<prometheus::Gauge>& vdvaus_unsupported_additional_runs_;
  prometheus::Family<prometheus::Gauge>& vdvaus_unsupported_additional_stops_;
  prometheus::Family<prometheus::Gauge>& vdvaus_current_matches_total_;
  prometheus::Family<prometheus::Gauge>& vdvaus_current_matches_non_empty_;
  prometheus::Family<prometheus::Gauge>& vdvaus_total_runs_;
  prometheus::Family<prometheus::Gauge>& vdvaus_complete_runs_;
  prometheus::Family<prometheus::Gauge>& vdvaus_unique_runs_;
  prometheus::Family<prometheus::Gauge>& vdvaus_matched_runs_;
  prometheus::Family<prometheus::Gauge>& vdvaus_multiple_matches_;
  prometheus::Family<prometheus::Gauge>& vdvaus_incomplete_not_seen_before_;
  prometheus::Family<prometheus::Gauge>& vdvaus_no_transport_found_at_stop_;
  prometheus::Family<prometheus::Gauge>& vdvaus_total_stops_;
  prometheus::Family<prometheus::Gauge>& vdvaus_resolved_stops_;
  prometheus::Family<prometheus::Gauge>& vdvaus_runs_without_stops_;
  prometheus::Family<prometheus::Gauge>& vdvaus_cancelled_runs_;
  prometheus::Family<prometheus::Gauge>& vdvaus_skipped_vdv_stops_;
  prometheus::Family<prometheus::Gauge>& vdvaus_excess_vdv_stops_;
  prometheus::Family<prometheus::Gauge>& vdvaus_updated_events_;
  prometheus::Family<prometheus::Gauge>& vdvaus_propagated_delays_;
  prometheus::Family<prometheus::Gauge>& vdvaus_feed_timestamp_;
  prometheus::Family<prometheus::Gauge>& vdvaus_last_update_timestamp_;
};

struct gtfsrt_metrics {
  explicit gtfsrt_metrics(std::string const& tag, rt_metric_families const& m)
      : updates_requested_{m.gtfsrt_updates_requested_.Add({{"tag", tag}})},
        updates_successful_{m.gtfsrt_updates_successful_.Add({{"tag", tag}})},
        updates_error_{m.gtfsrt_updates_error_.Add({{"tag", tag}})},
        total_entities_{m.gtfsrt_total_entities_.Add({{"tag", tag}})},
        total_entities_success_{
            m.gtfsrt_total_entities_success_.Add({{"tag", tag}})},
        total_entities_fail_{m.gtfsrt_total_entities_fail_.Add({{"tag", tag}})},
        unsupported_deleted_{m.gtfsrt_unsupported_deleted_.Add({{"tag", tag}})},
        unsupported_vehicle_{m.gtfsrt_unsupported_vehicle_.Add({{"tag", tag}})},
        unsupported_alert_{m.gtfsrt_unsupported_alert_.Add({{"tag", tag}})},
        unsupported_no_trip_id_{
            m.gtfsrt_unsupported_no_trip_id_.Add({{"tag", tag}})},
        no_trip_update_{m.gtfsrt_no_trip_update_.Add({{"tag", tag}})},
        trip_update_without_trip_{
            m.gtfsrt_trip_update_without_trip_.Add({{"tag", tag}})},
        trip_resolve_error_{m.gtfsrt_trip_resolve_error_.Add({{"tag", tag}})},
        unsupported_schedule_relationship_{
            m.gtfsrt_unsupported_schedule_relationship_.Add({{"tag", tag}})},
        feed_timestamp_{m.gtfsrt_feed_timestamp_.Add({{"tag", tag}})},
        last_update_timestamp_{
            m.gtfsrt_last_update_timestamp_.Add({{"tag", tag}})} {}

  void update(nigiri::rt::statistics const& stats) const {
    total_entities_.Increment(stats.total_entities_);
    total_entities_success_.Increment(stats.total_entities_success_);
    total_entities_fail_.Increment(stats.total_entities_fail_);
    unsupported_deleted_.Increment(stats.unsupported_deleted_);
    unsupported_vehicle_.Increment(stats.unsupported_vehicle_);
    unsupported_no_trip_id_.Increment(stats.unsupported_no_trip_id_);
    no_trip_update_.Increment(stats.no_trip_update_);
    trip_update_without_trip_.Increment(stats.trip_update_without_trip_);
    trip_resolve_error_.Increment(stats.trip_resolve_error_);
    unsupported_schedule_relationship_.Increment(
        stats.unsupported_schedule_relationship_);
    feed_timestamp_.Set(
        static_cast<double>(stats.feed_timestamp_.time_since_epoch().count()));
  }

  prometheus::Counter& updates_requested_;
  prometheus::Counter& updates_successful_;
  prometheus::Counter& updates_error_;

  prometheus::Counter& total_entities_;
  prometheus::Counter& total_entities_success_;
  prometheus::Counter& total_entities_fail_;
  prometheus::Counter& unsupported_deleted_;
  prometheus::Counter& unsupported_vehicle_;
  prometheus::Counter& unsupported_alert_;
  prometheus::Counter& unsupported_no_trip_id_;
  prometheus::Counter& no_trip_update_;
  prometheus::Counter& trip_update_without_trip_;
  prometheus::Counter& trip_resolve_error_;
  prometheus::Counter& unsupported_schedule_relationship_;
  prometheus::Gauge& feed_timestamp_;
  prometheus::Gauge& last_update_timestamp_;
};

struct vdvaus_metrics {
  explicit vdvaus_metrics(std::string const& tag, rt_metric_families const& m)
      : updates_requested_{m.vdvaus_updates_requested_.Add({{"tag", tag}})},
        updates_successful_{m.vdvaus_updates_successful_.Add({{"tag", tag}})},
        updates_error_{m.vdvaus_updates_error_.Add({{"tag", tag}})},
        unsupported_additional_runs_{
            m.vdvaus_unsupported_additional_runs_.Add({{"tag", tag}})},
        unsupported_additional_stops_{
            m.vdvaus_unsupported_additional_stops_.Add({{"tag", tag}})},
        current_matches_total_{
            m.vdvaus_current_matches_total_.Add({{"tag", tag}})},
        current_matches_non_empty_{
            m.vdvaus_current_matches_non_empty_.Add({{"tag", tag}})},
        total_runs_{m.vdvaus_total_runs_.Add({{"tag", tag}})},
        complete_runs_{m.vdvaus_complete_runs_.Add({{"tag", tag}})},
        unique_runs_{m.vdvaus_unique_runs_.Add({{"tag", tag}})},
        matched_runs_{m.vdvaus_matched_runs_.Add({{"tag", tag}})},
        multiple_matches_{m.vdvaus_multiple_matches_.Add({{"tag", tag}})},
        incomplete_not_seen_before_{
            m.vdvaus_incomplete_not_seen_before_.Add({{"tag", tag}})},
        no_transport_found_at_stop_{
            m.vdvaus_no_transport_found_at_stop_.Add({{"tag", tag}})},
        total_stops_{m.vdvaus_total_stops_.Add({{"tag", tag}})},
        resolved_stops_{m.vdvaus_resolved_stops_.Add({{"tag", tag}})},
        runs_without_stops_{m.vdvaus_runs_without_stops_.Add({{"tag", tag}})},
        cancelled_runs_{m.vdvaus_cancelled_runs_.Add({{"tag", tag}})},
        skipped_vdv_stops_{m.vdvaus_skipped_vdv_stops_.Add({{"tag", tag}})},
        excess_vdv_stops_{m.vdvaus_excess_vdv_stops_.Add({{"tag", tag}})},
        updated_events_{m.vdvaus_updated_events_.Add({{"tag", tag}})},
        propagated_delays_{m.vdvaus_propagated_delays_.Add({{"tag", tag}})},
        feed_timestamp_{m.vdvaus_feed_timestamp_.Add({{"tag", tag}})},
        last_update_timestamp_{
            m.vdvaus_last_update_timestamp_.Add({{"tag", tag}})} {}

  void update(nigiri::rt::vdv_aus::statistics const& stats) const {
    unsupported_additional_runs_.Set(stats.unsupported_additional_runs_);
    unsupported_additional_stops_.Set(stats.unsupported_additional_stops_);
    current_matches_total_.Set(
        static_cast<double>(stats.current_matches_total_));
    current_matches_non_empty_.Set(stats.current_matches_non_empty_);
    total_runs_.Set(stats.total_runs_);
    complete_runs_.Set(stats.complete_runs_);
    unique_runs_.Set(stats.unique_runs_);
    matched_runs_.Set(stats.matched_runs_);
    multiple_matches_.Set(stats.multiple_matches_);
    incomplete_not_seen_before_.Set(stats.incomplete_not_seen_before_);
    no_transport_found_at_stop_.Set(stats.no_transport_found_at_stop_);
    total_stops_.Set(stats.total_stops_);
    resolved_stops_.Set(stats.resolved_stops_);
    runs_without_stops_.Set(stats.runs_without_stops_);
    cancelled_runs_.Set(stats.cancelled_runs_);
    skipped_vdv_stops_.Set(stats.skipped_vdv_stops_);
    excess_vdv_stops_.Set(stats.excess_vdv_stops_);
    updated_events_.Set(stats.updated_events_);
    propagated_delays_.Set(stats.propagated_delays_);
  }

  prometheus::Counter& updates_requested_;
  prometheus::Counter& updates_successful_;
  prometheus::Counter& updates_error_;

  prometheus::Gauge& unsupported_additional_runs_;
  prometheus::Gauge& unsupported_additional_stops_;
  prometheus::Gauge& current_matches_total_;
  prometheus::Gauge& current_matches_non_empty_;
  prometheus::Gauge& total_runs_;
  prometheus::Gauge& complete_runs_;
  prometheus::Gauge& unique_runs_;
  prometheus::Gauge& matched_runs_;
  prometheus::Gauge& multiple_matches_;
  prometheus::Gauge& incomplete_not_seen_before_;
  prometheus::Gauge& no_transport_found_at_stop_;
  prometheus::Gauge& total_stops_;
  prometheus::Gauge& resolved_stops_;
  prometheus::Gauge& runs_without_stops_;
  prometheus::Gauge& cancelled_runs_;
  prometheus::Gauge& skipped_vdv_stops_;
  prometheus::Gauge& excess_vdv_stops_;
  prometheus::Gauge& updated_events_;
  prometheus::Gauge& propagated_delays_;
  prometheus::Gauge& feed_timestamp_;
  prometheus::Gauge& last_update_timestamp_;
};

}  // namespace motis