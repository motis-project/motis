#pragma once

#include <string>

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
        gtfsrt_source_events_{
            prometheus::BuildCounter()
                .Name("nigiri_gtfsrt_source_events_total")
                .Help("Last-good source cache and failure events")
                .Register(registry)},
        gtfsrt_source_state_{
            prometheus::BuildGauge()
                .Name("nigiri_gtfsrt_source_state")
                .Help("Current GTFS-RT source state (one-hot by state label)")
                .Register(registry)},
        gtfsrt_source_cache_age_{
            prometheus::BuildGauge()
                .Name("nigiri_gtfsrt_source_cache_age_seconds")
                .Help("Age of the materialized GTFS-RT source snapshot")
                .Register(registry)},
        gtfsrt_source_cache_fresh_{
            prometheus::BuildGauge()
                .Name("nigiri_gtfsrt_source_cache_fresh")
                .Help("Whether the materialized GTFS-RT snapshot is fresh")
                .Register(registry)},
        vehicle_eta_history_active_vehicles_{
            prometheus::BuildGauge()
                .Name("motis_vehicle_eta_history_active_vehicles")
                .Help("Vehicles retained in the published prediction history")
                .Register(registry)},
        vehicle_eta_history_observations_{
            prometheus::BuildGauge()
                .Name("motis_vehicle_eta_history_observations")
                .Help("Observations retained in prediction history")
                .Register(registry)},
        vehicle_eta_history_memory_bytes_{
            prometheus::BuildGauge()
                .Name("motis_vehicle_eta_history_memory_bytes")
                .Help("Estimated bytes retained by prediction history")
                .Register(registry)},
        vehicle_eta_history_update_seconds_{
            prometheus::BuildGauge()
                .Name("motis_vehicle_eta_history_update_seconds")
                .Help("CPU time spent updating prediction history this cycle")
                .Register(registry)},
        vehicle_eta_progress_outcomes_{
            prometheus::BuildGauge()
                .Name("motis_vehicle_eta_progress_outcomes")
                .Help(
                    "Current shadow progress outcomes by feed, mode and result")
                .Register(registry)},
        vehicle_eta_progress_lateral_error_meters_{
            prometheus::BuildGauge()
                .Name("motis_vehicle_eta_progress_lateral_error_meters")
                .Help("Current projected lateral error by feed, mode and "
                      "statistic")
                .Register(registry)},
        vehicle_eta_progress_evaluation_seconds_{
            prometheus::BuildGauge()
                .Name("motis_vehicle_eta_progress_evaluation_seconds")
                .Help(
                    "CPU time spent evaluating shadow trip progress this cycle")
                .Register(registry)},
        vehicle_eta_candidate_outcomes_{
            prometheus::BuildGauge()
                .Name("motis_vehicle_eta_candidate_outcomes")
                .Help("Current shadow ETA candidate outcomes by feed and mode")
                .Register(registry)},
        vehicle_eta_candidate_horizon_seconds_{
            prometheus::BuildGauge()
                .Name("motis_vehicle_eta_candidate_horizon_seconds")
                .Help("Current shadow ETA candidate horizon statistics")
                .Register(registry)},
        vehicle_eta_candidate_error_{
            prometheus::BuildGauge()
                .Name("motis_vehicle_eta_candidate_error")
                .Help("Current candidate versus provider raw-second and "
                      "rendered-minute error statistics")
                .Register(registry)},
        vehicle_eta_candidate_evaluation_seconds_{
            prometheus::BuildGauge()
                .Name("motis_vehicle_eta_candidate_evaluation_seconds")
                .Help("CPU time spent evaluating shadow ETA candidates")
                .Register(registry)},
        vehicle_eta_candidate_memory_bytes_{
            prometheus::BuildGauge()
                .Name("motis_vehicle_eta_candidate_memory_bytes")
                .Help("Estimated transient bytes used by shadow ETA results")
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
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_unsupported_additional_runs_total")
                .Help("Number of unsupported additional runs in the VDV AUS "
                      "feed")
                .Register(registry)},
        vdvaus_unsupported_additional_stops_{
            prometheus::BuildCounter()
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
        vdvaus_total_runs_{prometheus::BuildCounter()
                               .Name("nigiri_vdvaus_total_runs_total")
                               .Help("Total number of runs in the VDV AUS feed")
                               .Register(registry)},
        vdvaus_complete_runs_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_complete_runs_total")
                .Help("Total number of complete runs in the VDV AUS feed")
                .Register(registry)},
        vdvaus_unique_runs_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_unique_runs_total")
                .Help("Total number of unique runs in the VDV AUS feed")
                .Register(registry)},
        vdvaus_match_attempts_{prometheus::BuildCounter()
                                   .Name("nigiri_vdvaus_match_attempts_total")
                                   .Help("Total number of match attempts")
                                   .Register(registry)},
        vdvaus_matched_runs_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_matched_runs_total")
                .Help("Number of runs of the VDV AUS feed for which a "
                      "successful match attempt took place")
                .Register(registry)},
        vdvaus_found_runs_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_found_runs_total")
                .Help("Number of runs of the VDV AUS feed for which a matching "
                      "run in the static timetable could be looked up "
                      "successfully")
                .Register(registry)},
        vdvaus_multiple_matches_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_multiple_matches_total")
                .Help("Number of times a run of the VDV AUS feed could not be "
                      "matched to a transport in the timetable since there "
                      "were multiple transports with the same score")
                .Register(registry)},
        vdvaus_incomplete_not_seen_before_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_incomplete_not_seen_before_total")
                .Help(
                    "Number of times an incomplete run was encountered before "
                    "seeing a complete version of it in the VDV AUS feed")
                .Register(registry)},
        vdvaus_complete_after_incomplete_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_complete_after_incomplete_total")
                .Help("Number of times a complete run was encountered in the "
                      "feed after seeing an incomplete version before")
                .Register(registry)},
        vdvaus_no_transport_found_at_stop_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_no_transport_found_at_stop_total")
                .Help("Number of times that no transport could be found at the "
                      "stop specified in the VDV AUS feed")
                .Register(registry)},
        vdvaus_total_stops_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_total_stops_total")
                .Help("Total number of stops in the VDV AUS feed")
                .Register(registry)},
        vdvaus_resolved_stops_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_resolved_stops_total")
                .Help("Number of stops that could be resolved to locations in "
                      "the timetable")
                .Register(registry)},
        vdvaus_runs_without_stops_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_runs_without_stops_total")
                .Help("Number of times a run without any stops was encountered "
                      "in the VDV AUS feed")
                .Register(registry)},
        vdvaus_cancelled_runs_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_cancelled_runs_total")
                .Help("Number of cancelled runs in the VDV AUS feed")
                .Register(registry)},
        vdvaus_skipped_vdv_stops_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_skipped_vdv_stops_total")
                .Help("Number of stops in the VDV AUS feed that had to be "
                      "skipped while updating a run since they had no "
                      "counterpart in the run of the timetable")
                .Register(registry)},
        vdvaus_excess_vdv_stops_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_excess_vdv_stops_total")
                .Help(
                    "Number of additional stops at the end of runs in VDV AUS "
                    "feed that had no corresponding stop in the run of the "
                    "timetable that was updated")
                .Register(registry)},
        vdvaus_updated_events_{
            prometheus::BuildCounter()
                .Name("nigiri_vdvaus_updated_events_total")
                .Help("Number of arrival/departure times "
                      "that were updated by the VDV AUS feed")
                .Register(registry)},
        vdvaus_propagated_delays_{
            prometheus::BuildCounter()
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
  prometheus::Family<prometheus::Counter>& gtfsrt_source_events_;
  prometheus::Family<prometheus::Gauge>& gtfsrt_source_state_;
  prometheus::Family<prometheus::Gauge>& gtfsrt_source_cache_age_;
  prometheus::Family<prometheus::Gauge>& gtfsrt_source_cache_fresh_;
  prometheus::Family<prometheus::Gauge>& vehicle_eta_history_active_vehicles_;
  prometheus::Family<prometheus::Gauge>& vehicle_eta_history_observations_;
  prometheus::Family<prometheus::Gauge>& vehicle_eta_history_memory_bytes_;
  prometheus::Family<prometheus::Gauge>& vehicle_eta_history_update_seconds_;
  prometheus::Family<prometheus::Gauge>& vehicle_eta_progress_outcomes_;
  prometheus::Family<prometheus::Gauge>&
      vehicle_eta_progress_lateral_error_meters_;
  prometheus::Family<prometheus::Gauge>&
      vehicle_eta_progress_evaluation_seconds_;
  prometheus::Family<prometheus::Gauge>& vehicle_eta_candidate_outcomes_;
  prometheus::Family<prometheus::Gauge>& vehicle_eta_candidate_horizon_seconds_;
  prometheus::Family<prometheus::Gauge>& vehicle_eta_candidate_error_;
  prometheus::Family<prometheus::Gauge>&
      vehicle_eta_candidate_evaluation_seconds_;
  prometheus::Family<prometheus::Gauge>& vehicle_eta_candidate_memory_bytes_;

  prometheus::Family<prometheus::Counter>& vdvaus_updates_requested_;
  prometheus::Family<prometheus::Counter>& vdvaus_updates_successful_;
  prometheus::Family<prometheus::Counter>& vdvaus_updates_error_;

  prometheus::Family<prometheus::Counter>& vdvaus_unsupported_additional_runs_;
  prometheus::Family<prometheus::Counter>& vdvaus_unsupported_additional_stops_;
  prometheus::Family<prometheus::Gauge>& vdvaus_current_matches_total_;
  prometheus::Family<prometheus::Gauge>& vdvaus_current_matches_non_empty_;
  prometheus::Family<prometheus::Counter>& vdvaus_total_runs_;
  prometheus::Family<prometheus::Counter>& vdvaus_complete_runs_;
  prometheus::Family<prometheus::Counter>& vdvaus_unique_runs_;
  prometheus::Family<prometheus::Counter>& vdvaus_match_attempts_;
  prometheus::Family<prometheus::Counter>& vdvaus_matched_runs_;
  prometheus::Family<prometheus::Counter>& vdvaus_found_runs_;
  prometheus::Family<prometheus::Counter>& vdvaus_multiple_matches_;
  prometheus::Family<prometheus::Counter>& vdvaus_incomplete_not_seen_before_;
  prometheus::Family<prometheus::Counter>& vdvaus_complete_after_incomplete_;
  prometheus::Family<prometheus::Counter>& vdvaus_no_transport_found_at_stop_;
  prometheus::Family<prometheus::Counter>& vdvaus_total_stops_;
  prometheus::Family<prometheus::Counter>& vdvaus_resolved_stops_;
  prometheus::Family<prometheus::Counter>& vdvaus_runs_without_stops_;
  prometheus::Family<prometheus::Counter>& vdvaus_cancelled_runs_;
  prometheus::Family<prometheus::Counter>& vdvaus_skipped_vdv_stops_;
  prometheus::Family<prometheus::Counter>& vdvaus_excess_vdv_stops_;
  prometheus::Family<prometheus::Counter>& vdvaus_updated_events_;
  prometheus::Family<prometheus::Counter>& vdvaus_propagated_delays_;
  prometheus::Family<prometheus::Gauge>& vdvaus_feed_timestamp_;
  prometheus::Family<prometheus::Gauge>& vdvaus_last_update_timestamp_;
};

enum struct gtfsrt_source_state { no_base, live, replay, expired };

struct gtfsrt_metrics {
  explicit gtfsrt_metrics(std::string const& tag,
                          std::string const& endpoint,
                          rt_metric_families const& m)
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
            m.gtfsrt_last_update_timestamp_.Add({{"tag", tag}})},
        fetch_error_{m.gtfsrt_source_events_.Add(
            {{"tag", tag}, {"endpoint", endpoint}, {"event", "fetch_error"}})},
        empty_body_{m.gtfsrt_source_events_.Add(
            {{"tag", tag}, {"endpoint", endpoint}, {"event", "empty_body"}})},
        decode_error_{m.gtfsrt_source_events_.Add(
            {{"tag", tag}, {"endpoint", endpoint}, {"event", "decode_error"}})},
        missing_header_{
            m.gtfsrt_source_events_.Add({{"tag", tag},
                                         {"endpoint", endpoint},
                                         {"event", "missing_header"}})},
        last_good_reuse_{
            m.gtfsrt_source_events_.Add({{"tag", tag},
                                         {"endpoint", endpoint},
                                         {"event", "last_good_reuse"}})},
        last_good_expiry_{
            m.gtfsrt_source_events_.Add({{"tag", tag},
                                         {"endpoint", endpoint},
                                         {"event", "last_good_expiry"}})},
        recovery_{m.gtfsrt_source_events_.Add(
            {{"tag", tag}, {"endpoint", endpoint}, {"event", "recovery"}})},
        state_no_base_{m.gtfsrt_source_state_.Add(
            {{"tag", tag}, {"endpoint", endpoint}, {"state", "no_base"}})},
        state_live_{m.gtfsrt_source_state_.Add(
            {{"tag", tag}, {"endpoint", endpoint}, {"state", "live"}})},
        state_replay_{m.gtfsrt_source_state_.Add(
            {{"tag", tag}, {"endpoint", endpoint}, {"state", "replay"}})},
        state_expired_{m.gtfsrt_source_state_.Add(
            {{"tag", tag}, {"endpoint", endpoint}, {"state", "expired"}})},
        cache_age_{m.gtfsrt_source_cache_age_.Add(
            {{"tag", tag}, {"endpoint", endpoint}})},
        cache_fresh_{m.gtfsrt_source_cache_fresh_.Add(
            {{"tag", tag}, {"endpoint", endpoint}})} {
    set_source_state(gtfsrt_source_state::no_base, 0.0, false);
  }

  void set_source_state(gtfsrt_source_state const state,
                        double const cache_age,
                        bool const fresh) const {
    state_no_base_.Set(state == gtfsrt_source_state::no_base ? 1.0 : 0.0);
    state_live_.Set(state == gtfsrt_source_state::live ? 1.0 : 0.0);
    state_replay_.Set(state == gtfsrt_source_state::replay ? 1.0 : 0.0);
    state_expired_.Set(state == gtfsrt_source_state::expired ? 1.0 : 0.0);
    cache_age_.Set(cache_age);
    cache_fresh_.Set(fresh ? 1.0 : 0.0);
  }

  void update(nigiri::rt::statistics const& stats) const {
    total_entities_.Increment(stats.total_entities_);
    total_entities_success_.Increment(stats.total_entities_success_);
    total_entities_fail_.Increment(stats.total_entities_fail_);
    unsupported_deleted_.Increment(stats.unsupported_deleted_);
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
  prometheus::Counter& fetch_error_;
  prometheus::Counter& empty_body_;
  prometheus::Counter& decode_error_;
  prometheus::Counter& missing_header_;
  prometheus::Counter& last_good_reuse_;
  prometheus::Counter& last_good_expiry_;
  prometheus::Counter& recovery_;
  prometheus::Gauge& state_no_base_;
  prometheus::Gauge& state_live_;
  prometheus::Gauge& state_replay_;
  prometheus::Gauge& state_expired_;
  prometheus::Gauge& cache_age_;
  prometheus::Gauge& cache_fresh_;
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
        match_attempts_{m.vdvaus_match_attempts_.Add({{"tag", tag}})},
        matched_runs_{m.vdvaus_matched_runs_.Add({{"tag", tag}})},
        found_runs_{m.vdvaus_found_runs_.Add({{"tag", tag}})},
        multiple_matches_{m.vdvaus_multiple_matches_.Add({{"tag", tag}})},
        incomplete_not_seen_before_{
            m.vdvaus_incomplete_not_seen_before_.Add({{"tag", tag}})},
        complete_after_incomplete_{
            m.vdvaus_complete_after_incomplete_.Add({{"tag", tag}})},
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
        last_update_timestamp_{
            m.vdvaus_last_update_timestamp_.Add({{"tag", tag}})} {}

  void update(nigiri::rt::vdv_aus::statistics const& stats) const {
    unsupported_additional_runs_.Increment(stats.unsupported_additional_runs_);
    unsupported_additional_stops_.Increment(
        stats.unsupported_additional_stops_);
    current_matches_total_.Set(
        static_cast<double>(stats.current_matches_total_));
    current_matches_non_empty_.Set(stats.current_matches_non_empty_);
    total_runs_.Increment(stats.total_runs_);
    complete_runs_.Increment(stats.complete_runs_);
    unique_runs_.Increment(stats.unique_runs_);
    match_attempts_.Increment(stats.match_attempts_);
    matched_runs_.Increment(stats.matched_runs_);
    found_runs_.Increment(stats.found_runs_);
    multiple_matches_.Increment(stats.multiple_matches_);
    incomplete_not_seen_before_.Increment(stats.incomplete_not_seen_before_);
    complete_after_incomplete_.Increment(stats.complete_after_incomplete_);
    no_transport_found_at_stop_.Increment(stats.no_transport_found_at_stop_);
    total_stops_.Increment(stats.total_stops_);
    resolved_stops_.Increment(stats.resolved_stops_);
    runs_without_stops_.Increment(stats.runs_without_stops_);
    cancelled_runs_.Increment(stats.cancelled_runs_);
    skipped_vdv_stops_.Increment(stats.skipped_vdv_stops_);
    excess_vdv_stops_.Increment(stats.excess_vdv_stops_);
    updated_events_.Increment(stats.updated_events_);
    propagated_delays_.Increment(stats.propagated_delays_);
  }

  prometheus::Counter& updates_requested_;
  prometheus::Counter& updates_successful_;
  prometheus::Counter& updates_error_;

  prometheus::Counter& unsupported_additional_runs_;
  prometheus::Counter& unsupported_additional_stops_;
  prometheus::Gauge& current_matches_total_;
  prometheus::Gauge& current_matches_non_empty_;
  prometheus::Counter& total_runs_;
  prometheus::Counter& complete_runs_;
  prometheus::Counter& unique_runs_;
  prometheus::Counter& match_attempts_;
  prometheus::Counter& matched_runs_;
  prometheus::Counter& found_runs_;
  prometheus::Counter& multiple_matches_;
  prometheus::Counter& incomplete_not_seen_before_;
  prometheus::Counter& complete_after_incomplete_;
  prometheus::Counter& no_transport_found_at_stop_;
  prometheus::Counter& total_stops_;
  prometheus::Counter& resolved_stops_;
  prometheus::Counter& runs_without_stops_;
  prometheus::Counter& cancelled_runs_;
  prometheus::Counter& skipped_vdv_stops_;
  prometheus::Counter& excess_vdv_stops_;
  prometheus::Counter& updated_events_;
  prometheus::Counter& propagated_delays_;
  prometheus::Gauge& last_update_timestamp_;
};

}  // namespace motis
