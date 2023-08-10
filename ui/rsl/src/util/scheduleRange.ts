import {
  fromUnixTime,
  isAfter,
  isBefore,
  parseISO,
  startOfToday,
} from "date-fns";

import { LookupScheduleInfoResponse } from "@/api/protocol/motis/lookup";

export interface ScheduleRange {
  firstDay: string | undefined;
  lastDay: string | undefined;
  closestDay: string | undefined;
  closestDate: Date | undefined;
}

export function getScheduleRange(
  scheduleInfo: LookupScheduleInfoResponse | undefined,
): ScheduleRange {
  if (scheduleInfo == undefined) {
    return {
      firstDay: undefined,
      lastDay: undefined,
      closestDay: undefined,
      closestDate: undefined,
    };
  }

  // scheduleInfo.begin is 00:00 UTC at the first day of the loaded schedule
  // scheduleInfo.end is 00:00 UTC at the first day after the end of the loaded schedule

  // convert to calendar dates (YYYY-MM-DD format)
  const firstDay = fromUnixTime(scheduleInfo.begin)
    .toISOString()
    .substring(0, 10);

  // since scheduleInfo.end is 00:00 at the following day, we subtract an hour first
  const lastDay = fromUnixTime(scheduleInfo.end - 3600)
    .toISOString()
    .substring(0, 10);

  // these dates are 00:00 local time
  const localBegin = parseISO(firstDay);
  const localEnd = parseISO(lastDay);

  // 00:00 local time today
  const today = startOfToday();

  let closestDate = localBegin;
  if (!isBefore(today, localBegin) && !isAfter(today, localEnd)) {
    closestDate = today;
  }

  return {
    firstDay,
    lastDay,
    closestDay: closestDate.toISOString().substring(0, 10),
    closestDate,
  };
}
