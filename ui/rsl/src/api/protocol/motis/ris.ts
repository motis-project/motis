// GENERATED FILE - DO NOT MODIFY
// -> see /tools/protocol for information on how to update this file

// ris/RISContentType.fbs
export type RISContentType = "RIBasis" | "RISML";

// ris/RISForwardTimeRequest.fbs
export interface RISForwardTimeRequest {
  new_time: number;
  schedule: number;
}
