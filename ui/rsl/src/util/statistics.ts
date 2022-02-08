import {
  PaxMonCdfEntry,
  PaxMonTripLoadInfo,
} from "../api/protocol/motis/paxmon";

import {
  PaxMonEdgeLoadInfoWithStats,
  PaxMonTripLoadInfoWithStats,
} from "../data/loadInfo";

export function probPaxLE(cdf: PaxMonCdfEntry[], limit: number): number {
  let prob = 0.0;
  for (const e of cdf) {
    if (e.passengers > limit) {
      break;
    }
    prob = e.probability;
  }
  return prob;
}

export function probPaxGT(cdf: PaxMonCdfEntry[], limit: number): number {
  return Math.min(1.0, Math.max(0.0, 1.0 - probPaxLE(cdf, limit)));
}

export function paxQuantile(cdf: PaxMonCdfEntry[], q: number): number {
  let last = null;
  for (const e of cdf) {
    if (e.probability === q) {
      return e.passengers;
    } else if (e.probability > q) {
      if (last !== null) {
        return (last.passengers + e.passengers) / 2;
      } else {
        return e.passengers;
      }
    }
    last = e;
  }
  throw new Error("invalid cdf");
}

export function processEdgeForecast(
  ef: PaxMonEdgeLoadInfoWithStats
): PaxMonEdgeLoadInfoWithStats {
  ef.p_load_gt_100 =
    ef.capacity == 0 ? 0 : probPaxGT(ef.passenger_cdf, ef.capacity);
  ef.q_20 = paxQuantile(ef.passenger_cdf, 0.2);
  ef.q_50 = paxQuantile(ef.passenger_cdf, 0.5);
  ef.q_80 = paxQuantile(ef.passenger_cdf, 0.8);
  ef.q_5 = paxQuantile(ef.passenger_cdf, 0.05);
  ef.q_95 = paxQuantile(ef.passenger_cdf, 0.95);
  ef.min_pax = ef.passenger_cdf.length > 0 ? ef.passenger_cdf[0].passengers : 0;
  ef.max_pax =
    ef.passenger_cdf.length > 0
      ? ef.passenger_cdf[ef.passenger_cdf.length - 1].passengers
      : 0;
  return ef;
}

export function addEdgeStatistics(
  tripLoadInfo: PaxMonTripLoadInfo
): PaxMonTripLoadInfoWithStats {
  const withStats = tripLoadInfo as PaxMonTripLoadInfoWithStats;
  withStats.edges.forEach((e) => processEdgeForecast(e));
  return withStats;
}
