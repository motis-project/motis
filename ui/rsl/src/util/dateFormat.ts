import { format, fromUnixTime } from "date-fns";
import { de } from "date-fns/locale";

export function formatDateTime(ts: number): string {
  return format(fromUnixTime(ts), "dd.MM.yyyy HH:mm", {
    locale: de,
  });
}

export function formatLongDateTime(ts: number): string {
  return format(fromUnixTime(ts), "EEEE, dd.MM.yyyy, HH:mm O", {
    locale: de,
  });
}

export function formatDate(ts: number): string {
  return format(fromUnixTime(ts), "EEEE, dd.MM.yyyy", {
    locale: de,
  });
}

export function formatTime(ts: number): string {
  return format(fromUnixTime(ts), "HH:mm", {
    locale: de,
  });
}

export function formatFileNameTime(ts: number): string {
  return format(fromUnixTime(ts), "yyyy-MM-dd_HHmm");
}
