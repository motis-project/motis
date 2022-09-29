import { useAtom } from "jotai";

import { selectedTripAtom } from "@/data/selectedTrip";

import TripDetails from "@/components/trips/TripDetails";
import TripList from "@/components/trips/TripList";

function SelectedTripDetails(): JSX.Element {
  const [selectedTrip] = useAtom(selectedTripAtom);

  if (selectedTrip !== undefined) {
    return (
      <TripDetails
        tripId={selectedTrip.trip}
        key={JSON.stringify(selectedTrip.trip)}
      />
    );
  } else {
    return <></>;
  }
}

function TripsMainSection(): JSX.Element {
  return (
    <>
      <div className="bg-db-cool-gray-200 dark:bg-gray-800 w-[25rem] overflow-y-auto p-2 shrink-0">
        <TripList />
      </div>
      <div className="overflow-y-auto grow p-2">
        <SelectedTripDetails />
      </div>
    </>
  );
}

export default TripsMainSection;
