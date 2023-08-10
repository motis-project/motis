import { PaxMonCapacityData } from "@/api/protocol/motis/paxmon";

export const EMTPY_CAPACITY_DATA: PaxMonCapacityData = {
  limit: 0,
  seats: 0,
  seats_1st: 0,
  seats_2nd: 0,
  standing: 0,
  total_limit: 0,
};

export function addCapacityData(
  a: PaxMonCapacityData,
  b: PaxMonCapacityData,
): PaxMonCapacityData {
  return {
    limit: a.limit + b.limit,
    seats: a.seats + b.seats,
    seats_1st: a.seats_1st + b.seats_1st,
    seats_2nd: a.seats_2nd + b.seats_2nd,
    standing: a.standing + b.standing,
    total_limit: a.total_limit + b.total_limit,
  };
}
