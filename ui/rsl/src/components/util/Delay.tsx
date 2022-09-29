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
  let mins = Math.round(minutes);
  const color = getDelayColor(mins);

  let text = "";
  if (mins >= 60) {
    const hours = Math.floor(mins / 60);
    mins = mins % 60;
    text = `${hours}h${mins}m`;
  } else {
    text = `${mins}m`;
  }

  return <span className={color}>{text}</span>;
}

export default Delay;
