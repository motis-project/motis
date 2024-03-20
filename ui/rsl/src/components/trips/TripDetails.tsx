import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useAtom } from "jotai";
import { ExternalLink } from "lucide-react";
import React, { ReactNode } from "react";
import { useParams } from "react-router-dom";

import { TripId } from "@/api/protocol/motis";

import {
  queryKeys,
  sendPaxMonGetTripLoadInfosRequest,
  usePaxMonStatusQuery,
} from "@/api/paxmon.ts";

import { universeAtom } from "@/data/multiverse.ts";
import { showLegacyLoadForecastChartAtom } from "@/data/settings";
import { activeTripTabAtom } from "@/data/views.ts";

import { getBahnTrainSearchUrl } from "@/util/bahnDe.ts";
import { formatDate } from "@/util/dateFormat.ts";

import CapacityInfo from "@/components/trips/CapacityInfo";
import { CheckData } from "@/components/trips/CheckData.tsx";
import TripLoadForecastChart from "@/components/trips/TripLoadForecastChart";
import TripRoute from "@/components/trips/TripRoute";
import { TripTransfers } from "@/components/trips/TripTransfers.tsx";
import { Button } from "@/components/ui/button.tsx";
import { Skeleton } from "@/components/ui/skeleton.tsx";
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
  const [universe] = useAtom(universeAtom);
  const { data: paxmonStatus } = usePaxMonStatusQuery(universe);
  const showCheckData =
    paxmonStatus != undefined && paxmonStatus.has_check_data;

  return (
    <div>
      <TripTitle tripId={tripId} />
      <Tabs
        defaultValue={activeTripTab}
        onValueChange={setActiveTripTab}
        className="mt-2 text-center"
      >
        <TabsList>
          <TabsTrigger value="forecast">Auslastungsprognose</TabsTrigger>
          <TabsTrigger value="transfers">Anschlüsse</TabsTrigger>
          <TabsTrigger value="capacity">Kapazitätsdaten</TabsTrigger>
          {showCheckData && (
            <TabsTrigger value="check">Reisendenzähldaten</TabsTrigger>
          )}
        </TabsList>
        <TabsContent value="forecast" className="text-left">
          <TripRoute tripId={tripId} />
          {showLegacyLoadForecastChart && (
            <TripLoadForecastChart tripId={tripId} mode="Interactive" />
          )}
        </TabsContent>
        <TabsContent value="transfers" className="text-left">
          <TripTransfers tripId={tripId} />
        </TabsContent>
        <TabsContent value="capacity" className="text-left">
          <CapacityInfo tripId={tripId} />
        </TabsContent>
        {showCheckData && (
          <TabsContent value="check" className="text-left">
            <CheckData tripId={tripId} />
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
}

interface TripTitleProps {
  tripId: TripId;
}

function TripTitle({ tripId }: TripTitleProps) {
  const [universe] = useAtom(universeAtom);
  const queryClient = useQueryClient();
  const { data /*, isLoading, error*/ } = useQuery({
    queryKey: queryKeys.tripLoad(universe, tripId),
    queryFn: () =>
      sendPaxMonGetTripLoadInfosRequest({ universe, trips: [tripId] }),
    placeholderData: () => {
      return universe != 0
        ? queryClient.getQueryData(queryKeys.tripLoad(0, tripId))
        : undefined;
    },
  });
  const tsi = data?.load_infos[0]?.tsi;
  const line = tsi?.service_infos[0]?.line;
  const [, ...secondaryServices] = tsi?.service_infos ?? [];

  return (
    <>
      <div className="my-2 flex items-center justify-center gap-6 text-lg">
        <div className="text-2xl font-medium">
          {tsi ? (
            `${tsi.service_infos[0]?.category ?? ""} ${tsi.service_infos[0]?.train_nr ?? tsi.trip.train_nr}`
          ) : (
            <Skeleton className="h-10 w-24" />
          )}
        </div>
        {line && <div>Linie {line}</div>}
        <div>
          {tsi ? formatDate(tsi.trip.time) : <Skeleton className="h-10 w-40" />}
        </div>
        <div className="flex items-center gap-2">
          {tsi ? (
            <>
              {" "}
              {tsi.primary_station.name}
              <span>→</span>
              {tsi.secondary_station.name}
            </>
          ) : (
            <Skeleton className="h-10 w-80" />
          )}
        </div>
        {tsi && (
          <Button variant="outline" className="gap-2" asChild>
            <a
              href={getBahnTrainSearchUrl(tsi.trip, tsi.primary_station)}
              target="_blank"
              referrerPolicy="no-referrer"
              rel="noreferrer"
            >
              <ExternalLink className="h-4 w-4" aria-hidden="true" />
              Zugverlauf auf bahn.de
            </a>
          </Button>
        )}
      </div>
      {secondaryServices.length > 0 && (
        <div className="flex items-center justify-center gap-3">
          <span>Fährt teilweise auch als:</span>
          {secondaryServices.map((si, idx) => (
            <span key={idx}>{`${si.category} ${si.train_nr}`}</span>
          ))}
        </div>
      )}
    </>
  );
}

export function TripDetailsFromRoute(): ReactNode {
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
