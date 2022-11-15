export function formatShortDuration(minutes: number): string {
  let mins = Math.round(minutes);
  let text = "";
  if (mins >= 60) {
    const hours = Math.floor(mins / 60);
    mins = mins % 60;
    text = `${hours}h${mins}m`;
  } else {
    text = `${mins}m`;
  }
  return text;
}
