import { useAtom } from "jotai";
import { useState } from "react";

import { TripId } from "@/api/protocol/motis";

import { useLookupRiBasisQuery } from "@/api/lookup";

import { PaxMonEdgeLoadInfoWithStats } from "@/data/loadInfo";
import { scheduleAtom } from "@/data/simulation";

import TripLoadForecastChart from "@/components/TripLoadForecastChart";
import TripSectionDetails from "@/components/TripSectionDetails";

type TripDetailsProps = {
  tripId: TripId;
};

function TripDetails({ tripId }: TripDetailsProps): JSX.Element {
  const [selectedSection, setSelectedSection] =
    useState<PaxMonEdgeLoadInfoWithStats>();
  const [schedule] = useAtom(scheduleAtom);

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
