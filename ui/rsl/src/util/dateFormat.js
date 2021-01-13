export const dateTimeFormat = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "long",
});

export const timeFormat = new Intl.DateTimeFormat(undefined, {
  timeStyle: "short",
});

export function formatDateTime(ts) {
  return dateTimeFormat.format(new Date(ts * 1000));
}

export function formatTime(ts) {
  return timeFormat.format(new Date(ts * 1000));
}
