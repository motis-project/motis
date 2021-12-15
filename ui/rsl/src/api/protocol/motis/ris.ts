// generated file - do not modify - run update-protocol to update

// ris/RISContentType.fbs
export type RISContentType = "RIBasis" | "RISML";

// ris/RISForwardTimeRequest.fbs
export interface RISForwardTimeRequest {
  new_time: number;
  schedule: number;
}
