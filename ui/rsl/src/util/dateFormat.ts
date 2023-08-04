import { format, formatISO, fromUnixTime, parse } from "date-fns";
import { de } from "date-fns/locale";

function getDate(ts: Date | number): Date {
  return typeof ts === "number" ? fromUnixTime(ts) : ts;
}

export function formatDateTime(ts: Date | number): string {
  return format(getDate(ts), "dd.MM.yyyy HH:mm", {
    locale: de,
  });
}

export function formatShortDateTime(ts: Date | number): string {
  return format(getDate(ts), "dd.MM. HH:mm", {
    locale: de,
  });
}

export function formatLongDateTime(ts: Date | number): string {
  return format(getDate(ts), "EEEE, dd.MM.yyyy, HH:mm O", {
    locale: de,
  });
}

export function formatDate(ts: Date | number): string {
  return format(getDate(ts), "EEEE, dd.MM.yyyy", {
    locale: de,
  });
}

export function formatISODate(ts: Date | number): string {
  return formatISO(getDate(ts), { representation: "date" });
}

export function formatTime(ts: Date | number, fmt = "HH:mm"): string {
  return format(getDate(ts), fmt, {
    locale: de,
  });
}

export function formatFileNameTime(ts: Date | number): string {
  return format(getDate(ts), "yyyy-MM-dd_HHmm");
}

const RI_BASIS_TIMESTAMP_FORMAT = "yyyy-MM-dd'T'HH:mm:ssxxx";

export function formatRiBasisDateTime(ts: Date): string {
  return format(ts, RI_BASIS_TIMESTAMP_FORMAT);
}

export function parseRiBasisDateTime(str: string, referenceDate?: Date): Date {
  return parse(str, RI_BASIS_TIMESTAMP_FORMAT, referenceDate ?? new Date());
}
