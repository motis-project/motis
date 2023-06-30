// GENERATED FILE - DO NOT MODIFY
// -> see /tools/protocol for information on how to update this file

// rt/RtMetricsRequest.fbs
export interface RtMetricsRequest {
  schedule: number;
}

// rt/RtMetricsResponse.fbs
export interface RtMetrics {
  start_time: number;
  entries: number;
  messages: number[];
  delay_messages: number[];
  cancel_messages: number[];
  additional_messages: number[];
  reroute_messages: number[];
  track_messages: number[];
  full_trip_messages: number[];
  trip_formation_messages: number[];
  full_trip_schedule_messages: number[];
  full_trip_update_messages: number[];
  new_trips: number[];
  cancellations: number[];
  reroutes: number[];
  rule_service_reroutes: number[];
  trip_delay_updates: number[];
  event_delay_updates: number[];
  trip_track_updates: number[];
  trip_id_not_found: number[];
  trip_id_ambiguous: number[];
  formation_schedule_messages: number[];
  formation_preview_messages: number[];
  formation_is_messages: number[];
  formation_invalid_trip_id: number[];
  formation_trip_id_not_found: number[];
  formation_trip_id_ambiguous: number[];
}

// rt/RtMetricsResponse.fbs
export interface RtMetricsResponse {
  by_msg_timestamp: RtMetrics;
  by_processing_time: RtMetrics;
}
