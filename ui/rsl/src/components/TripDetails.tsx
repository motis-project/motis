import { useAtom } from "jotai";
import { useState } from "react";

import { useLookupRiBasisQuery } from "../api/lookup";
import { TripId } from "../api/protocol/motis";

import { PaxMonEdgeLoadInfoWithStats } from "../data/loadInfo";
import { scheduleAtom } from "../data/simulation";

import TripLoadForecastChart from "./TripLoadForecastChart";
import TripSectionDetails from "./TripSectionDetails";

type TripDetailsProps = {
  tripId: TripId;
};

function TripDetails({ tripId }: TripDetailsProps): JSX.Element {
  const [selectedSection, setSelectedSection] =
    useState<PaxMonEdgeLoadInfoWithStats>();
  const [schedule] = useAtom(scheduleAtom);
  const { data: ribData } = useLookupRiBasisQuery({
    trip_id: tripId,
    schedule,
  });

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
      <div>
        <p>RI Basis:</p>
        <textarea
          value={ribData ? JSON.stringify(ribData, null, 2) : ""}
          readOnly={true}
          rows={20}
          className="
                    mt-1
                    block
                    w-full
                    rounded-md
                    border-gray-300
                    shadow-sm
                    focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50
                  "
        />
      </div>
    </div>
  );
}

export default TripDetails;
