import {
  PaxMonCdfEntry,
  PaxMonTripLoadInfo,
} from "@/api/protocol/motis/paxmon";

import {
  PaxMonEdgeLoadInfoWithStats,
  PaxMonTripLoadInfoWithStats,
} from "@/data/loadInfo";

export function probPaxLE(cdf: PaxMonCdfEntry[], limit: number): number {
  let prob = 0.0;
  for (const e of cdf) {
    if (e.n > limit) {
      break;
    }
    prob = e.p;
  }
  return prob;
}

export function probPaxGT(cdf: PaxMonCdfEntry[], limit: number): number {
  return Math.min(1.0, Math.max(0.0, 1.0 - probPaxLE(cdf, limit)));
}

export function paxQuantile(cdf: PaxMonCdfEntry[], q: number): number {
  let last = null;
  for (const e of cdf) {
    if (e.p === q) {
      return e.n;
    } else if (e.p > q) {
      if (last !== null) {
        return (last.n + e.n) / 2;
      } else {
        return e.n;
      }
    }
    last = e;
  }
  throw new Error("invalid cdf");
}

export function getMaxPax(cdf: PaxMonCdfEntry[]): number {
  return cdf[cdf.length - 1]?.n ?? 0;
}

export function processEdgeForecast(
  ef: PaxMonEdgeLoadInfoWithStats
): PaxMonEdgeLoadInfoWithStats {
  // TODO: REMOVE

  /*
  ef.p_load_gt_100 = ef.capacity == 0 ? 0 : probPaxGT(ef.dist.cdf, ef.capacity);
  ef.q_50 = paxQuantile(ef.dist.cdf, 0.5);
  ef.q_5 = paxQuantile(ef.dist.cdf, 0.05);
  ef.q_95 = paxQuantile(ef.dist.cdf, 0.95);
  ef.min_pax = ef.dist.cdf.length > 0 ? ef.dist.cdf[0].n : 0;
  ef.max_pax =
    ef.dist.cdf.length > 0 ? ef.dist.cdf[ef.dist.cdf.length - 1].n : 0;
   */

  ef.p_load_gt_100 = ef.prob_over_capacity;
  ef.q_50 = ef.dist.q50;
  ef.q_5 = ef.dist.q5;
  ef.q_95 = ef.dist.q95;
  ef.min_pax = ef.dist.min;
  ef.max_pax = ef.dist.max;
  return ef;
}

export function addEdgeStatistics(
  tripLoadInfo: PaxMonTripLoadInfo
): PaxMonTripLoadInfoWithStats {
  const withStats = tripLoadInfo as PaxMonTripLoadInfoWithStats;
  withStats.edges.forEach((e) => processEdgeForecast(e));
  return withStats;
}
