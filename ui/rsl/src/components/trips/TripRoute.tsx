import {
  ChevronDownIcon,
  ChevronRightIcon,
  ExclamationTriangleIcon,
  QuestionMarkCircleIcon,
} from "@heroicons/react/20/solid";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useAtom, useSetAtom } from "jotai";
import React, { ReactNode, useEffect, useState } from "react";

import { TripId } from "@/api/protocol/motis";
import { PaxMonEdgeLoadInfo } from "@/api/protocol/motis/paxmon";

import {
  queryKeys,
  sendPaxMonGetTripLoadInfosRequest,
  usePaxMonStatusQuery,
} from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";
import { formatPercent } from "@/data/numberFormat";
import { mostRecentlySelectedTripAtom } from "@/data/selectedTrip";
import { sectionGraphPlotTypeAtom } from "@/data/settings";

import {
  getCapacitySourceTooltip,
  isExactCapacitySource,
} from "@/util/capacitySource";
import { SectionLoadColors } from "@/util/colors";
import { formatTime } from "@/util/dateFormat";

import SectionLoadGraph from "@/components/trips/SectionLoadGraph";
import TripOptimization from "@/components/trips/TripOptimization";
import TripSectionDetails from "@/components/trips/TripSectionDetails";

import { cn } from "@/lib/utils";

interface TripRouteProps {
  tripId: TripId;
}

function TripRoute({ tripId }: TripRouteProps): ReactNode {
  const [universe] = useAtom(universeAtom);
  const { data: status } = usePaxMonStatusQuery(universe);

  const queryClient = useQueryClient();
  const { data /*, isLoading, error*/ } = useQuery({
    queryKey: queryKeys.tripLoad(universe, tripId),
    queryFn: () =>
      sendPaxMonGetTripLoadInfosRequest({ universe, trips: [tripId] }),
    enabled: !!status,
    placeholderData: () => {
      return universe != 0
        ? queryClient.getQueryData(queryKeys.tripLoad(0, tripId))
        : undefined;
    },
  });

  const setMostRecentlySelectedTrip = useSetAtom(mostRecentlySelectedTripAtom);
  useEffect(() => {
    if (data && data.load_infos.length > 0) {
      setMostRecentlySelectedTrip(data.load_infos[0].tsi);
    }
  }, [data, setMostRecentlySelectedTrip]);

  if (!data) {
    return <div className="w-full text-center">Zugverlauf wird geladen...</div>;
  }

  const tripData = data.load_infos[0];
  const edges = tripData.edges;
  const sectionCount = tripData.edges.length;

  const maxPax = edges.reduce((max, ef) => Math.max(max, ef.dist.max), 0);
  const maxExpected = edges.reduce(
    (max, ef) => Math.max(max, ef.expected_passengers),
    0,
  );
  const maxCapacity = edges.reduce(
    (max, ef) => (ef.capacity ? Math.max(max, ef.capacity) : max),
    0,
  );
  const maxVal = Math.max(maxPax, maxExpected, maxCapacity);
  const missingExactCapacityInfo = edges.some(
    (eli) => !isExactCapacitySource(eli.capacity_source),
  );

  const optimizationAvailable = edges.some((e) => e.possibly_over_capacity);

  return (
    <div>
      <TripOptimization
        tripId={tripId}
        optimizationAvailable={optimizationAvailable}
      />
      <div className="flex flex-col gap-2">
        {edges.map((section, idx) => (
          <TripSection
            key={idx}
            tripId={tripId}
            section={section}
            index={idx}
            sectionCount={sectionCount}
            maxVal={maxVal}
            showCapacitySource={missingExactCapacityInfo}
          />
        ))}
        <Legend />
      </div>
    </div>
  );
}

interface TripSectionProps {
  tripId: TripId;
  section: PaxMonEdgeLoadInfo;
  index: number;
  sectionCount: number;
  maxVal: number;
  showCapacitySource: boolean;
}

function TripSection({
  tripId,
  section,
  maxVal,
  showCapacitySource,
}: TripSectionProps) {
  const [expanded, setExpanded] = useState(false);
  const [sectionGraphPlotType] = useAtom(sectionGraphPlotTypeAtom);

  const departureDelayed =
    section.departure_current_time > section.departure_schedule_time;
  const arrivalDelayed =
    section.arrival_current_time > section.arrival_schedule_time;

  const ChevronIcon = expanded ? ChevronDownIcon : ChevronRightIcon;

  return (
    <>
      <div
        className="group flex cursor-pointer gap-1"
        onClick={() => setExpanded((val) => !val)}
      >
        <div>
          <ChevronIcon className="mt-0.5 h-5 w-5 group-hover:fill-db-red-500" />
        </div>
        <div className="w-80 shrink-0">
          <div className="flex gap-1">
            <div className="flex w-24 shrink-0 justify-between">
              <span className="w-1/2 text-gray-600">
                {formatTime(section.departure_schedule_time)}
              </span>
              <span
                className={cn(
                  "w-1/2",
                  departureDelayed ? "text-red-600" : "text-green-600",
                )}
              >
                {formatTime(section.departure_current_time)}
              </span>
            </div>
            <div className="truncate hover:relative hover:overflow-visible">
              {section.from.name}
            </div>
          </div>
          <div className="flex gap-1">
            <div className="flex w-24 shrink-0 justify-between">
              <span className="w-1/2 text-gray-600">
                {formatTime(section.arrival_schedule_time)}
              </span>
              <span
                className={cn(
                  "w-1/2",
                  arrivalDelayed ? "text-red-600" : "text-green-600",
                )}
              >
                {formatTime(section.arrival_current_time)}
              </span>
            </div>
            <div className="truncate hover:relative hover:overflow-visible">
              {section.to.name}
            </div>
          </div>
        </div>
        <div
          className="flex w-10 flex-col items-center pt-1"
          title={`Überlastungswahrscheinlichkeit: ${formatPercent(
            section.prob_over_capacity,
          )}`}
        >
          {section.prob_over_capacity >= 0.01 ? (
            <>
              <span>
                <ExclamationTriangleIcon className="h-5 w-5 fill-db-red-500" />
              </span>
              <span className="text-xs text-db-red-500">
                {formatPercent(section.prob_over_capacity)}
              </span>
            </>
          ) : null}
        </div>
        <div className="shrink grow">
          <div className="h-16 w-full">
            <SectionLoadGraph
              section={section}
              maxVal={maxVal}
              plotType={sectionGraphPlotType}
            />
          </div>
        </div>
        {showCapacitySource && (
          <div
            className="flex w-7 justify-center pt-3"
            title={getCapacitySourceTooltip(section.capacity_source)}
          >
            {!isExactCapacitySource(section.capacity_source) ? (
              <QuestionMarkCircleIcon className="h-5 w-5 fill-db-cool-gray-500" />
            ) : null}
          </div>
        )}
      </div>
      {expanded ? (
        <TripSectionDetails tripId={tripId} selectedSection={section} />
      ) : null}
    </>
  );
}

function Legend() {
  const [sectionGraphPlotType] = useAtom(sectionGraphPlotTypeAtom);

  return (
    <div>
      <div className="flex flex-wrap justify-end gap-x-5 gap-y-2 pb-2 pr-2 pt-2 text-sm">
        <div>Auslastungsstufen:</div>
        <div className="flex items-center gap-1">
          <span
            className="inline-block h-5 w-5 rounded-md"
            style={{ backgroundColor: SectionLoadColors.Bg_0_80 }}
          />
          &lt;80%
        </div>
        <div className="flex items-center gap-1">
          <span
            className="inline-block h-5 w-5 rounded-md"
            style={{ backgroundColor: SectionLoadColors.Bg_80_100 }}
          />
          80-100%
        </div>
        <div className="flex items-center gap-1">
          <span
            className="inline-block h-5 w-5 rounded-md"
            style={{ backgroundColor: SectionLoadColors.Bg_100_120 }}
          />
          100-120%
        </div>
        <div className="flex items-center gap-1">
          <span
            className="inline-block h-5 w-5 rounded-md"
            style={{ backgroundColor: SectionLoadColors.Bg_120_200 }}
          />
          120-200%
        </div>
        <div className="flex items-center gap-1">
          <span
            className="inline-block h-5 w-5 rounded-md"
            style={{ backgroundColor: SectionLoadColors.Bg_200_plus }}
          />
          &gt;200%
        </div>
      </div>
      <div className="flex flex-wrap justify-end gap-x-5 gap-y-2 pb-4 pr-2 pt-2 text-sm">
        {sectionGraphPlotType == "SimpleBox" ? (
          <div className="flex items-center gap-1">
            <svg width={20} height={20} viewBox="0 0 20 20">
              <rect
                x={0}
                y={0}
                width={20}
                height={20}
                rx={5}
                fill={SectionLoadColors.Fill_Range}
              />
              <path
                d="M10 0 V20"
                stroke={SectionLoadColors.Stroke_Median}
                strokeWidth={3}
                fill="none"
              />
            </svg>
            Prognostizierte Auslastung (Spannbreite und Median)
          </div>
        ) : (
          <div className="flex items-center gap-1">
            <svg width={20} height={20} viewBox="0 0 20 20">
              <rect
                x={0}
                y={0}
                width={20}
                height={20}
                rx={6}
                className="fill-db-cool-gray-200"
              />
              <rect
                x={4}
                y={4}
                width={12}
                height={12}
                rx={4}
                fill={SectionLoadColors.Fill_BoxViolin}
                stroke={SectionLoadColors.Stroke_BoxViolin}
                strokeWidth={2}
              />
            </svg>
            Prognostizierte Auslastung (
            {sectionGraphPlotType == "Box" ? "Box-Plot" : "Violin-Plot"})
          </div>
        )}
        <div className="flex items-center gap-1">
          <svg width={20} height={20} viewBox="0 0 20 20">
            <rect
              x={0}
              y={0}
              width={20}
              height={20}
              rx={6}
              className="fill-db-cool-gray-200"
            />
            <path
              d="M10 0 V20"
              stroke={SectionLoadColors.Stroke_Expected1}
              strokeDasharray={2}
              strokeWidth={2}
              fill="none"
            />
            <path
              d="M10 2 V20"
              stroke={SectionLoadColors.Stroke_Expected2}
              strokeDasharray={2}
              strokeWidth={2}
              fill="none"
            />
          </svg>
          Planmäßige Auslastung
        </div>
      </div>
    </div>
  );
}

export default TripRoute;
