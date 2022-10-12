import { useAtom } from "jotai";

import { selectedTripAtom } from "@/data/selectedTrip";

import classNames from "@/util/classNames";

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

type TripsMainSectionProps = {
  visible?: boolean;
};

function TripsMainSection({
  visible = true,
}: TripsMainSectionProps): JSX.Element {
  const visibilityClass = visible ? "block" : "hidden";
  return (
    <>
      <div
        className={classNames(
          visibilityClass,
          "bg-db-cool-gray-200 dark:bg-gray-800 w-[25rem] overflow-y-auto p-2 shrink-0"
        )}
      >
        <TripList />
      </div>
      <div className={classNames(visibilityClass, "overflow-y-auto grow p-2")}>
        <SelectedTripDetails />
      </div>
    </>
  );
}

export default TripsMainSection;
