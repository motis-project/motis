// GENERATED FILE - DO NOT MODIFY
// -> see /tools/protocol for information on how to update this file

// ris/RISContentType.fbs
export type RISContentType = "RIBasis" | "RISML";

// ris/RISForwardTimeRequest.fbs
export interface RISForwardTimeRequest {
  new_time: number;
  schedule: number;
}

// ris/RISStatusResponse.fbs
export interface RISSourceStatus {
  enabled: boolean;
  update_interval: number;
  last_update_time: number;
  last_update_messages: number;
  last_message_time: number;
  total_updates: number;
  total_messages: number;
}

// ris/RISStatusResponse.fbs
export interface RISStatusResponse {
  system_time: number;
  last_update_time: number;
  gtfs_rt_status: RISSourceStatus;
  ribasis_fahrt_status: RISSourceStatus;
  ribasis_formation_status: RISSourceStatus;
  upload_status: RISSourceStatus;
  read_status: RISSourceStatus;
  init_status: RISSourceStatus;
}
