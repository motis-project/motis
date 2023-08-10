import { formatShortDuration } from "@/data/durationFormat";

interface DelayProps {
  minutes: number;
  colored?: boolean;
  forceSign?: boolean;
}

function getDelayColor(mins: number): string {
  if (mins < 6) {
    return "text-green-600";
  } else if (mins < 20) {
    return "text-amber-600";
  } else if (mins < 60) {
    return "text-orange-600";
  } else if (mins < 120) {
    return "text-red-600";
  } else {
    return "text-pink-600";
  }
}

function Delay({
  minutes,
  colored = true,
  forceSign = false,
}: DelayProps): JSX.Element {
  return (
    <span className={colored ? getDelayColor(Math.round(minutes)) : ""}>
      {formatShortDuration(minutes, forceSign)}
    </span>
  );
}

export default Delay;
