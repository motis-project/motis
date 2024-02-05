import { useAtom } from "jotai";
import React, { ReactNode } from "react";
import { useParams } from "react-router-dom";

import { TripId } from "@/api/protocol/motis";

import { showLegacyLoadForecastChartAtom } from "@/data/settings";
import { activeTripTabAtom } from "@/data/views.ts";

import CapacityInfo from "@/components/trips/CapacityInfo";
import { CheckData } from "@/components/trips/CheckData.tsx";
import TripLoadForecastChart from "@/components/trips/TripLoadForecastChart";
import TripRoute from "@/components/trips/TripRoute";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs.tsx";

interface TripDetailsProps {
  tripId: TripId;
}

function TripDetails({ tripId }: TripDetailsProps): ReactNode {
  const [showLegacyLoadForecastChart] = useAtom(
    showLegacyLoadForecastChartAtom,
  );
  const [activeTripTab, setActiveTripTab] = useAtom(activeTripTabAtom);

  return (
    <div>
      <Tabs
        defaultValue={activeTripTab}
        onValueChange={setActiveTripTab}
        className="mt-2 text-center"
      >
        <TabsList>
          <TabsTrigger value="forecast">Auslastungsprognose</TabsTrigger>
          <TabsTrigger value="capacity">Kapazitätsdaten</TabsTrigger>
          <TabsTrigger value="check">Reisendenzähldaten</TabsTrigger>
        </TabsList>
        <TabsContent value="forecast" className="text-left">
          <TripRoute tripId={tripId} />
          {showLegacyLoadForecastChart && (
            <TripLoadForecastChart tripId={tripId} mode="Interactive" />
          )}
        </TabsContent>
        <TabsContent value="capacity" className="text-left">
          <CapacityInfo tripId={tripId} />
        </TabsContent>
        <TabsContent value="check" className="text-left">
          <CheckData tripId={tripId} />
        </TabsContent>
      </Tabs>
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
