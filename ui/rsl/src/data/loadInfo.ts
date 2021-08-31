import { PaxMonEdgeLoadInfo } from "../api/protocol/motis/paxmon";
import { TripServiceInfo } from "../api/protocol/motis";

export interface PaxMonEdgeLoadInfoWithStats extends PaxMonEdgeLoadInfo {
  p_load_gt_100: number;
  q_20: number;
  q_50: number;
  q_80: number;
  q_5: number;
  q_95: number;
  min_pax: number;
  max_pax: number;
}

export interface PaxMonTripLoadInfoWithStats {
  tsi: TripServiceInfo;
  edges: PaxMonEdgeLoadInfoWithStats[];
}
