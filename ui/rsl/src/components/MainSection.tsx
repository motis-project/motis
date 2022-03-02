import { useAtom } from "jotai";

import { selectedTripAtom } from "@/data/selectedTrip";

import TripDetails from "@/components/TripDetails";

function MainSection(): JSX.Element {
  const [selectedTrip] = useAtom(selectedTripAtom);

  return (
    <>
      {selectedTrip !== undefined ? (
        <TripDetails tripId={selectedTrip} />
      ) : null}
    </>
  );
}

export default MainSection;
