export function formatShortDuration(
  minutes: number,
  forceSign = false,
): string {
  let mins = Math.round(minutes);
  let text = "";
  if (mins >= 60) {
    const hours = Math.floor(mins / 60);
    mins = mins % 60;
    text = `${hours}h${mins}m`;
  } else if (mins <= -60) {
    const hours = Math.ceil(mins / 60);
    mins = mins % 60;
    text = `${hours}h${-mins}m`;
  } else {
    text = `${mins}m`;
  }
  return forceSign && mins >= 0 ? `+${text}` : text;
}
