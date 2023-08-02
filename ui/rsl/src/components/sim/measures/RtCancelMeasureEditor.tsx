import { useQueryClient } from "@tanstack/react-query";
import { PrimitiveAtom, useAtom } from "jotai";
import { focusAtom } from "jotai-optics";
import { useAtomCallback } from "jotai/utils";
import { zip } from "lodash-es";
import React, { useCallback, useEffect, useMemo, useState } from "react";

import { TripServiceInfo } from "@/api/protocol/motis";
import {
  RiBasisFahrtData,
  RiBasisHaltestelle,
} from "@/api/protocol/motis/ribasis";

import {
  queryKeys as lookupQueryKeys,
  sendLookupRiBasisRequest,
} from "@/api/lookup";

import {
  MeasureUnion,
  RtCancelMeasureData,
  isRtCancelMeasureU,
} from "@/data/measures";
import { scheduleAtom } from "@/data/multiverse";

import TripPicker from "@/components/inputs/TripPicker";

import { cn } from "@/lib/utils";

export interface RtCancelMeasureEditorProps {
  measureAtom: PrimitiveAtom<MeasureUnion>;
  closeEditor: () => void;
  deleteMeasure: (measureAtom: PrimitiveAtom<MeasureUnion>) => void;
}

const labelClass = "font-semibold";

function RtCancelMeasureEditor({
  measureAtom,
  closeEditor,
  deleteMeasure,
}: RtCancelMeasureEditorProps): JSX.Element {
  const dataAtom = useMemo(
    () =>
      focusAtom(measureAtom, (optic) =>
        optic.guard(isRtCancelMeasureU).prop("data"),
      ),
    [measureAtom],
  );
  const [data, setData] = useAtom(dataAtom);
  const queryClient = useQueryClient();
  const [selectedTrip, setSelectedTrip] = useState<string>();

  if (!data) {
    throw new Error("invalid measure editor");
  }

  const getSchedule = useAtomCallback(
    useCallback((get) => {
      return get(scheduleAtom);
    }, []),
  );

  const setTrip = async (tsi: TripServiceInfo | undefined) => {
    setData((d) => {
      return { ...d, trip: tsi };
    });
    if (tsi) {
      const schedule = getSchedule();
      const lookupReq = { trip_id: tsi.trip, schedule };
      const data = await queryClient.fetchQuery(
        lookupQueryKeys.riBasis(lookupReq),
        () => sendLookupRiBasisRequest(lookupReq),
        { staleTime: 1000 },
      );
      console.log(`received ri basis: ${data.trips.length} trips`);
      const requestedTripId = JSON.stringify(tsi.trip);
      const tripData = data.trips.filter(
        (t) => JSON.stringify(t.trip_id) === requestedTripId,
      );
      if (tripData.length === 1) {
        setSelectedTrip(requestedTripId);
        setData((d) => {
          const fahrtData = tripData[0].fahrt.data;
          return {
            ...d,
            original_ribasis: fahrtData,
            canceled_stops: Array<boolean>(
              fahrtData.allFahrtabschnitt.length + 1,
            ).fill(false),
            allow_reroute: data.trips.length === 1,
          };
        });
      } else {
        console.log("no ri basis for requested trip received");
      }
    }
  };

  useEffect(() => {
    if (!data.original_ribasis && data.trip) {
      setTrip(data.trip).catch((err) =>
        console.log("RtCancelMeasureEditor init failed:", err),
      );
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div>
      <div className="mb-4">
        <div className={labelClass}>Zug</div>
        <div>
          <TripPicker
            onTripPicked={setTrip}
            clearOnPick={false}
            longDistanceOnly={false}
            initialTrip={data.trip}
          />
        </div>
      </div>
      {data.original_ribasis ? (
        <div>
          <StopListEditor
            data={data}
            setData={setData}
            key={selectedTrip}
            closeEditor={closeEditor}
            measureAtom={measureAtom}
            deleteMeasure={deleteMeasure}
          />
        </div>
      ) : null}
    </div>
  );
}

interface StopInfo {
  stop: RiBasisHaltestelle;
}

function getStops(ribasis: RiBasisFahrtData | undefined): StopInfo[] {
  const stops: StopInfo[] = [];
  if (!ribasis) {
    return stops;
  }
  for (const section of ribasis.allFahrtabschnitt) {
    stops.push({ stop: section.abfahrt.haltestelle });
  }
  if (ribasis.allFahrtabschnitt.length > 0) {
    const lastSection =
      ribasis.allFahrtabschnitt[ribasis.allFahrtabschnitt.length - 1];
    stops.push({ stop: lastSection.ankunft.haltestelle });
  }
  return stops;
}

interface StopListEditorProps {
  data: RtCancelMeasureData;
  setData: React.Dispatch<React.SetStateAction<RtCancelMeasureData>>;
  closeEditor: () => void;
  measureAtom: PrimitiveAtom<MeasureUnion>;
  deleteMeasure: (measureAtom: PrimitiveAtom<MeasureUnion>) => void;
}

function StopListEditor({
  data,
  setData,
  closeEditor,
  measureAtom,
  deleteMeasure,
}: StopListEditorProps): JSX.Element {
  const stops = useMemo(
    () => getStops(data.original_ribasis),
    [data.original_ribasis],
  );

  const allCanceled = data.canceled_stops.every((c) => c);

  const toggleStop = useCallback(
    (idx: number) => {
      setData((d) => {
        const cs = d.canceled_stops;
        cs[idx] = !cs[idx];
        return { ...d, canceled_stops: cs };
      });
    },
    [setData],
  );

  const toggleAll = useCallback(() => {
    setData((d) => {
      const current = d.canceled_stops.every((c) => c);
      return {
        ...d,
        canceled_stops: Array<boolean>(d.canceled_stops.length).fill(!current),
      };
    });
  }, [setData]);

  if (!data.allow_reroute) {
    return (
      <div>
        <div className="font-semibold">
          Dieser Zug ist an Vereinigungen und/oder Durchbindungen beteiligt.
          (Teil-)Ausfälle werden momentan nicht unterstützt.
        </div>
        <div className="pt-5 flex flex-col">
          <button
            onClick={() => deleteMeasure(measureAtom)}
            className="mt-4 px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white rounded"
          >
            Abbrechen
          </button>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className={labelClass}>Ausgefallene Halte:</div>
      <div className="mt-2 mb-4">
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={allCanceled}
            onChange={toggleAll}
            className="rounded border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-offset-0 focus:ring-blue-200 focus:ring-opacity-50"
          />
          <span className={cn(allCanceled && "line-through")}>Alle Halte</span>
        </label>
      </div>
      <div className="flex flex-col gap-2">
        {zip(stops, data.canceled_stops).map(([stop, canceled], idx) => (
          <div key={idx}>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={canceled}
                onChange={() => toggleStop(idx)}
                className="rounded border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-offset-0 focus:ring-blue-200 focus:ring-opacity-50"
              />
              <span className={cn(canceled && "line-through")}>
                {stop?.stop?.bezeichnung}
              </span>
            </label>
          </div>
        ))}
      </div>
      <div className="pt-5 flex flex-col">
        <button
          onClick={() => closeEditor()}
          className="px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white rounded"
        >
          Maßnahme speichern
        </button>
      </div>
    </div>
  );
}

export default RtCancelMeasureEditor;
