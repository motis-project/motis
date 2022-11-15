import { formatShortDuration } from "@/data/durationFormat";

type DelayProps = {
  minutes: number;
};

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

function Delay({ minutes }: DelayProps): JSX.Element {
  return (
    <span className={getDelayColor(Math.round(minutes))}>
      {formatShortDuration(minutes)}
    </span>
  );
}

export default Delay;
