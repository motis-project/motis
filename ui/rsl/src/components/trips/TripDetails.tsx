import { useAtom } from "jotai";
import { useParams } from "react-router-dom";

import { TripId } from "@/api/protocol/motis";

import { showLegacyLoadForecastChartAtom } from "@/data/settings";

import TripLoadForecastChart from "@/components/trips/TripLoadForecastChart";
import TripRoute from "@/components/trips/TripRoute";

type TripDetailsProps = {
  tripId: TripId;
};

function TripDetails({ tripId }: TripDetailsProps): JSX.Element {
  const [showLegacyLoadForecastChart] = useAtom(
    showLegacyLoadForecastChartAtom
  );

  return (
    <div>
      <TripRoute tripId={tripId} />
      {showLegacyLoadForecastChart && (
        <TripLoadForecastChart tripId={tripId} mode="Interactive" />
      )}
    </div>
  );
}

export function TripDetailsFromRoute(): JSX.Element {
  const params = useParams();
  // TODO: validate
  const tripId = params.tripId
    ? (JSON.parse(params.tripId) as TripId)
    : undefined;
  if (tripId !== undefined) {
    return <TripDetails tripId={tripId} key={JSON.stringify(tripId)} />;
  } else {
    return <></>;
  }
}

export default TripDetails;
