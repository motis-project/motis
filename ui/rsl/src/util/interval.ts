import { add, getUnixTime } from "date-fns";

import { Interval } from "@/api/protocol/motis";

export function getDayInterval(date: Date | undefined | null): Interval {
  return {
    begin: date ? getUnixTime(date) : 0,
    end: date ? getUnixTime(add(date, { days: 1 })) : 0,
  };
}
