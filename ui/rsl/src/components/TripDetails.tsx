import { useState } from "react";

import { TripId } from "@/api/protocol/motis";

import { PaxMonEdgeLoadInfoWithStats } from "@/data/loadInfo";

import TripLoadForecastChart from "@/components/TripLoadForecastChart";
import TripSectionDetails from "@/components/TripSectionDetails";

type TripDetailsProps = {
  tripId: TripId;
};

function TripDetails({ tripId }: TripDetailsProps): JSX.Element {
  const [selectedSection, setSelectedSection] =
    useState<PaxMonEdgeLoadInfoWithStats>();

  return (
    <div>
      <TripLoadForecastChart
        tripId={tripId}
        mode="Interactive"
        onSectionClick={setSelectedSection}
      />
      {selectedSection && (
        <TripSectionDetails
          tripId={tripId}
          selectedSection={selectedSection}
          onClose={() => setSelectedSection(undefined)}
        />
      )}
    </div>
  );
}

export default TripDetails;
