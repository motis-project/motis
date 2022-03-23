import { TripId } from "@/api/protocol/motis";

import TripLoadForecastChart from "@/components/TripLoadForecastChart";
import TripRoute from "@/components/TripRoute";

type TripDetailsProps = {
  tripId: TripId;
};

function TripDetails({ tripId }: TripDetailsProps): JSX.Element {
  return (
    <div>
      <TripRoute tripId={tripId} />
      <TripLoadForecastChart tripId={tripId} mode="Interactive" />
    </div>
  );
}

export default TripDetails;
