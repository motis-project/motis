import { useAtom } from "jotai";
import { useParams } from "react-router-dom";

import { TripId } from "@/api/protocol/motis";

import {
  showCapacityInfoAtom,
  showLegacyLoadForecastChartAtom,
} from "@/data/settings";

import CapacityInfo from "@/components/trips/CapacityInfo";
import TripLoadForecastChart from "@/components/trips/TripLoadForecastChart";
import TripRoute from "@/components/trips/TripRoute";

interface TripDetailsProps {
  tripId: TripId;
}

function TripDetails({ tripId }: TripDetailsProps): JSX.Element {
  const [showLegacyLoadForecastChart] = useAtom(
    showLegacyLoadForecastChartAtom,
  );
  const [showCapacityInfo] = useAtom(showCapacityInfoAtom);

  return (
    <div>
      <TripRoute tripId={tripId} />
      {showCapacityInfo && <CapacityInfo tripId={tripId} />}
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
