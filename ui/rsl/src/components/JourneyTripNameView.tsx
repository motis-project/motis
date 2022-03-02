import { JourneyTrip } from "@/data/journey";

type JourneyTripNameViewProps = {
  jt: JourneyTrip;
};

function JourneyTripNameView({ jt }: JourneyTripNameViewProps): JSX.Element {
  const names = [
    ...new Set(jt.transports.map((t) => `${t.train_nr} [${t.name}]`)),
  ];
  return <span>{names.join(", ")}</span>;
}

export default JourneyTripNameView;
