import { PaxMonCdfEntry } from "@/api/protocol/motis/paxmon";

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
