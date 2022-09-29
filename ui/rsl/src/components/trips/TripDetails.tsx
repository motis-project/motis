import { useAtom } from "jotai";

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

export default TripDetails;
