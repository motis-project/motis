import {
  ChevronDownIcon,
  ChevronRightIcon,
  ExclamationIcon,
} from "@heroicons/react/solid";
import { useAtom } from "jotai";
import { useState } from "react";
import { useQuery, useQueryClient } from "react-query";

import { TripId } from "@/api/protocol/motis";
import { PaxMonEdgeLoadInfo } from "@/api/protocol/motis/paxmon";

import {
  queryKeys,
  sendPaxMonGetTripLoadInfosRequest,
  usePaxMonStatusQuery,
} from "@/api/paxmon";

import { formatPercent } from "@/data/numberFormat";
import { universeAtom } from "@/data/simulation";

import classNames from "@/util/classNames";
import { formatDate, formatTime } from "@/util/dateFormat";

import SectionLoadGraph from "@/components/SectionLoadGraph";
import TripSectionDetails from "@/components/TripSectionDetails";

type TripRouteProps = {
  tripId: TripId;
};

function TripRoute({ tripId }: TripRouteProps): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const { data: status } = usePaxMonStatusQuery(universe);

  const queryClient = useQueryClient();
  const { data /*, isLoading, error*/ } = useQuery(
    queryKeys.tripLoad(universe, tripId),
    () => sendPaxMonGetTripLoadInfosRequest({ universe, trips: [tripId] }),
    {
      enabled: !!status,
      placeholderData: () => {
        return universe != 0
          ? queryClient.getQueryData(queryKeys.tripLoad(0, tripId))
          : undefined;
      },
    }
  );

  if (!data) {
    return <div className="w-full text-center">Zugverlauf wird geladen...</div>;
  }

  const tripData = data.load_infos[0];
  const edges = tripData.edges;
  const sectionCount = tripData.edges.length;

  const maxPax = edges.reduce((max, ef) => Math.max(max, ef.dist.max), 0);
  const maxExpected = edges.reduce(
    (max, ef) => Math.max(max, ef.expected_passengers),
    0
  );
  const maxCapacity = edges.reduce(
    (max, ef) => (ef.capacity ? Math.max(max, ef.capacity) : max),
    0
  );
  const maxVal = Math.max(maxPax, maxExpected, maxCapacity);

  const category = tripData.tsi.service_infos[0]?.category ?? "";
  const trainNr =
    tripData.tsi.service_infos[0]?.train_nr ?? tripData.tsi.trip.train_nr;
  const line = tripData.tsi.service_infos[0]?.line;

  return (
    <div className="">
      <div className="mb-4 flex gap-6 items-center text-lg justify-center">
        <span className="font-medium text-2xl">
          {category} {trainNr}
        </span>
        {line && <span>Linie {line}</span>}
        <span>{formatDate(tripData.tsi.trip.time)}</span>
        <span>
          {tripData.tsi.primary_station.name} →{" "}
          {tripData.tsi.secondary_station.name}
        </span>
      </div>
      <div className="flex flex-col gap-2">
        {edges.map((section, idx) => (
          <TripSection
            key={idx}
            tripId={tripId}
            section={section}
            index={idx}
            sectionCount={sectionCount}
            maxVal={maxVal}
          />
        ))}
      </div>
    </div>
  );
}

type TripSectionProps = {
  tripId: TripId;
  section: PaxMonEdgeLoadInfo;
  index: number;
  sectionCount: number;
  maxVal: number;
};

function TripSection({ tripId, section, maxVal }: TripSectionProps) {
  const [expanded, setExpanded] = useState(false);

  const departureDelayed =
    section.departure_current_time > section.departure_schedule_time;
  const arrivalDelayed =
    section.arrival_current_time > section.arrival_schedule_time;

  const ChevronIcon = expanded ? ChevronDownIcon : ChevronRightIcon;

  return (
    <>
      <div
        className="flex gap-1 cursor-pointer group"
        onClick={() => setExpanded((val) => !val)}
      >
        <div>
          <ChevronIcon className="w-5 h-5 mt-0.5 group-hover:fill-db-red-500" />
        </div>
        <div className="w-80">
          <div className="flex gap-1">
            <div className="flex justify-between w-24 shrink-0">
              <span className="text-gray-600 w-1/2">
                {formatTime(section.departure_schedule_time)}
              </span>
              <span
                className={classNames(
                  "w-1/2",
                  departureDelayed ? "text-red-600" : "text-green-600"
                )}
              >
                {formatTime(section.departure_current_time)}
              </span>
            </div>
            <div className="truncate hover:overflow-visible hover:relative">
              {section.from.name}
            </div>
          </div>
          <div className="flex gap-1">
            <div className="flex justify-between w-24 shrink-0">
              <span className="text-gray-600 w-1/2">
                {formatTime(section.arrival_schedule_time)}
              </span>
              <span
                className={classNames(
                  "w-1/2",
                  arrivalDelayed ? "text-red-600" : "text-green-600"
                )}
              >
                {formatTime(section.arrival_current_time)}
              </span>
            </div>
            <div className="truncate hover:overflow-visible hover:relative">
              {section.to.name}
            </div>
          </div>
        </div>
        <div
          className="w-10 pt-1 flex flex-col items-center"
          title={`Überlastungswahrscheinlichkeit: ${formatPercent(
            section.prob_over_capacity
          )}`}
        >
          {section.prob_over_capacity >= 0.01 ? (
            <>
              <span>
                <ExclamationIcon className="w-5 h-5 fill-db-red-500" />
              </span>
              <span className="text-xs text-db-red-500">
                {formatPercent(section.prob_over_capacity)}
              </span>
            </>
          ) : null}
        </div>
        <div className="flex-grow">
          <div className="w-full h-16">
            <SectionLoadGraph section={section} maxVal={maxVal} />
          </div>
        </div>
      </div>
      {expanded ? (
        <TripSectionDetails
          tripId={tripId}
          selectedSection={section}
          onClose={() => setExpanded(false)}
        />
      ) : null}
    </>
  );
}

export default TripRoute;
