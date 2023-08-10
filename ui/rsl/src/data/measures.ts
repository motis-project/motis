import { PrimitiveAtom, atom } from "jotai";
import { v4 as uuidv4 } from "uuid";

import { Station, TripServiceInfo } from "@/api/protocol/motis";
import {
  LoadLevel,
  MeasureRecipients,
  MeasureType,
  MeasureWrapper,
  OverrideCapacitySection,
} from "@/api/protocol/motis/paxforecast";
import {
  RiBasisFahrt,
  RiBasisFahrtAbschnitt,
  RiBasisFahrtData,
} from "@/api/protocol/motis/ribasis";

import { formatRiBasisDateTime } from "@/util/dateFormat";

export interface MeasureRecipientsData {
  trips: TripServiceInfo[];
  stations: Station[];
}

export interface SharedMeasureData {
  recipients: MeasureRecipientsData;
  time: Date;
}

export interface TripLoadInfoMeasureData {
  trip: TripServiceInfo | undefined;
  level: LoadLevel;
}

export interface TripRecommendationMeasureData {
  planned_destination: Station | undefined;
  recommended_trip: TripServiceInfo | undefined;
}

export interface TripLoadRecommendationMeasureData {
  planned_destination: Station | undefined;
  full_trip: TripLoadInfoMeasureData;
  recommended_trips: TripLoadInfoMeasureData[];
}

export interface RtUpdateMeasureData {
  trip: TripServiceInfo | undefined;
  ribasis: RiBasisFahrtData | undefined;
}

export interface RtCancelMeasureData {
  trip: TripServiceInfo | undefined;
  original_ribasis: RiBasisFahrtData | undefined;
  canceled_stops: boolean[];
  allow_reroute: boolean;
}

export interface UpdateCapacityMeasureData {
  trip: TripServiceInfo | undefined;
  seats: number;
}

export type UiMeasureType = MeasureType | "Empty" | "RtCancelMeasure";

export interface EmptyMeasureU {
  type: "Empty";
  shared: SharedMeasureData;
}

export interface TripLoadInfoMeasureU {
  type: "TripLoadInfoMeasure";
  shared: SharedMeasureData;
  data: TripLoadInfoMeasureData;
}

export interface TripRecommendationMeasureU {
  type: "TripRecommendationMeasure";
  shared: SharedMeasureData;
  data: TripRecommendationMeasureData;
}

export interface TripLoadRecommendationMeasureU {
  type: "TripLoadRecommendationMeasure";
  shared: SharedMeasureData;
  data: TripLoadRecommendationMeasureData;
}

export interface RtUpdateMeasureU {
  type: "RtUpdateMeasure";
  shared: SharedMeasureData;
  data: RtUpdateMeasureData;
}

export interface RtCancelMeasureU {
  type: "RtCancelMeasure";
  shared: SharedMeasureData;
  data: RtCancelMeasureData;
}

export interface UpdateCapacitiesMeasureU {
  type: "UpdateCapacitiesMeasure";
  shared: SharedMeasureData;
  data: UpdateCapacityMeasureData;
}

export type MeasureUnion =
  | EmptyMeasureU
  | TripLoadInfoMeasureU
  | TripRecommendationMeasureU
  | TripLoadRecommendationMeasureU
  | RtUpdateMeasureU
  | RtCancelMeasureU
  | UpdateCapacitiesMeasureU;

export function isEmptyMeasureU(mu: MeasureUnion): mu is EmptyMeasureU {
  return mu.type === "Empty";
}

export function isTripLoadInfoMeasureU(
  mu: MeasureUnion,
): mu is TripLoadInfoMeasureU {
  return mu.type === "TripLoadInfoMeasure";
}

export function isTripRecommendationMeasureU(
  mu: MeasureUnion,
): mu is TripRecommendationMeasureU {
  return mu.type === "TripRecommendationMeasure";
}

export function isTripLoadRecommendationMeasureU(
  mu: MeasureUnion,
): mu is TripLoadRecommendationMeasureU {
  return mu.type === "TripLoadRecommendationMeasure";
}

export function isRtUpdateMeasureU(mu: MeasureUnion): mu is RtUpdateMeasureU {
  return mu.type === "RtUpdateMeasure";
}

export function isRtCancelMeasureU(mu: MeasureUnion): mu is RtCancelMeasureU {
  return mu.type === "RtCancelMeasure";
}

export function measureSupportsRecipients(mu: MeasureUnion): boolean {
  return mu.type !== "UpdateCapacitiesMeasure";
}

export function measureNeedsRecipients(mu: MeasureUnion): boolean {
  return measureTypeNeedsRecipients(mu.type);
}

export function measureTypeNeedsRecipients(
  type: MeasureUnion["type"],
): boolean {
  return (
    type !== "RtUpdateMeasure" &&
    type !== "RtCancelMeasure" &&
    type !== "UpdateCapacitiesMeasure"
  );
}

export function isUpdateCapacitiesMeasureU(
  mu: MeasureUnion,
): mu is UpdateCapacitiesMeasureU {
  return mu.type === "UpdateCapacitiesMeasure";
}

export function toMeasureWrapper(mu: MeasureUnion): MeasureWrapper | null {
  const shared = {
    recipients: {
      trips: mu.shared.recipients.trips.map((t) => t.trip),
      stations: mu.shared.recipients.stations.map((s) => s.id),
    },
    time: Math.round(mu.shared.time.getTime() / 1000),
  };

  switch (mu.type) {
    case "Empty":
      return null;
    case "TripLoadInfoMeasure": {
      const d = mu.data;
      if (!d.trip) {
        return null;
      }
      return {
        measure_type: "TripLoadInfoMeasure",
        measure: { ...shared, trip: d.trip.trip, level: d.level },
      };
    }
    case "TripRecommendationMeasure": {
      const d = mu.data;
      if (!d.planned_destination || !d.recommended_trip) {
        return null;
      }
      return {
        measure_type: "TripRecommendationMeasure",
        measure: {
          ...shared,
          planned_trips: [],
          planned_destinations: [d.planned_destination.id],
          recommended_trip: d.recommended_trip.trip,
        },
      };
    }
    case "TripLoadRecommendationMeasure": {
      const d = mu.data;
      const recommendedTrips = d.recommended_trips
        .filter(
          (tll): tll is { trip: TripServiceInfo; level: LoadLevel } =>
            tll.trip != undefined,
        )
        .map((tll) => {
          return { trip: tll.trip.trip, level: tll.level };
        });
      if (
        !d.planned_destination ||
        !d.full_trip.trip ||
        recommendedTrips.length == 0
      ) {
        return null;
      }
      return {
        measure_type: "TripLoadRecommendationMeasure",
        measure: {
          ...shared,
          planned_destinations: [d.planned_destination.id],
          full_trips: [
            { trip: d.full_trip.trip.trip, level: d.full_trip.level },
          ],
          recommended_trips: recommendedTrips,
        },
      };
    }
    case "RtUpdateMeasure": {
      const d = mu.data;
      if (!d.ribasis) {
        return null;
      }
      const ribf = makeRiBasisFahrt(d.ribasis, mu.shared.time);
      return makeRtUpdateMeasure(shared, ribf);
    }
    case "RtCancelMeasure": {
      const d = mu.data;
      if (!d.original_ribasis || d.canceled_stops.every((c) => !c)) {
        return null;
      }
      const updated = cancelStops(d.original_ribasis, d.canceled_stops);
      const ribf = makeRiBasisFahrt(updated, mu.shared.time);
      return makeRtUpdateMeasure(shared, ribf);
    }
    case "UpdateCapacitiesMeasure": {
      const d = mu.data;
      if (!d.trip) {
        return null;
      }
      const sections: OverrideCapacitySection[] = [];
      if (d.seats !== 0) {
        sections.push({
          departure_station: "",
          departure_schedule_time: 0,
          seats: d.seats,
        });
      }
      return {
        measure_type: "OverrideCapacityMeasure",
        measure: {
          time: shared.time,
          trip: d.trip.trip,
          sections,
        },
      };
    }
  }
}

function makeRiBasisFahrt(data: RiBasisFahrtData, time: Date): RiBasisFahrt {
  return {
    meta: {
      id: uuidv4(),
      owner: "",
      format: "RIPL",
      version: "v3",
      correlation: [],
      created: formatRiBasisDateTime(time),
      sequence: time.getTime(),
    },
    data,
  };
}

function makeRtUpdateMeasure(
  shared: { recipients: MeasureRecipients; time: number },
  ribasis: RiBasisFahrt,
): MeasureWrapper {
  return {
    measure_type: "RtUpdateMeasure",
    measure: {
      ...shared,
      type: "RIBasis",
      content: JSON.stringify(ribasis, null, 2),
    },
  };
}

function cancelStops(
  original: RiBasisFahrtData,
  canceledStops: boolean[],
): RiBasisFahrtData {
  const sections: RiBasisFahrtAbschnitt[] = [];
  let lastDeparture: RiBasisFahrtAbschnitt | null = null;

  for (const [idx, sec] of original.allFahrtabschnitt.entries()) {
    const depCanceled = canceledStops[idx];
    const arrCanceled = canceledStops[idx + 1];
    if (depCanceled && arrCanceled) {
      continue;
    } else if (!depCanceled && !arrCanceled) {
      sections.push(sec);
      lastDeparture = null;
    } else if (arrCanceled) {
      lastDeparture = sec;
    } else if (depCanceled) {
      if (lastDeparture) {
        sections.push({ ...lastDeparture, ankunft: sec.ankunft });
        lastDeparture = null;
      }
    }
  }

  return { ...original, allFahrtabschnitt: sections };
}

export function newEmptyMeasure(time: Date): MeasureUnion {
  return {
    type: "Empty",
    shared: { recipients: { trips: [], stations: [] }, time },
  };
}

export const measuresAtom = atom<PrimitiveAtom<MeasureUnion>[]>([]);
export const currentEditorMeasureAtom =
  atom<PrimitiveAtom<MeasureUnion> | null>(null);
