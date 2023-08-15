import { JourneyTrip } from "@/data/journey";

interface JourneyTripNameViewProps {
  jt: JourneyTrip;
}

function JourneyTripNameView({ jt }: JourneyTripNameViewProps): JSX.Element {
  const names = [...new Set(jt.transports.map((t) => `${t.name}`))];
  // return <span>{names.join(", ")}</span>;
  return <span>{names.length > 0 ? names[0] : "???"}</span>;
}

export default JourneyTripNameView;
