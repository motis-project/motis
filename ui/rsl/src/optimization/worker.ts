import { fromUnixTime } from "date-fns";
import { partition, zipWith } from "lodash-es";

import { TripId, TripServiceInfo } from "@/api/protocol/motis";
import { LoadLevel } from "@/api/protocol/motis/paxforecast";
import {
  PaxMonEdgeLoadInfo,
  PaxMonGetAddressableGroupsResponse,
  PaxMonGetGroupsInTripResponse,
  PaxMonGroupBaseInfo,
  PaxMonTripLoadInfo,
} from "@/api/protocol/motis/paxmon";

import { setApiEndpoint } from "@/api/endpoint";
import {
  sendPaxMonAddressableGroupsRequest,
  sendPaxMonDestroyUniverseRequest,
  sendPaxMonForkUniverseRequest,
  sendPaxMonGetTripLoadInfosRequest,
  sendPaxMonGroupsInTripRequest,
} from "@/api/paxmon";
import { sendRoutingRequest } from "@/api/routing";

import {
  Journey,
  TripLeg,
  connectionToJourney,
  getArrivalTime,
} from "@/data/journey";
import { MeasureUnion, SharedMeasureData } from "@/data/measures";

import {
  OptimizationType,
  WorkerRequest,
  WorkerUpdate,
} from "@/optimization/workerMessages";

/////
const SEARCH_INTERVAL = 61;
/////

function log(msg: string) {
  postMessage({ type: "Log", msg } as WorkerUpdate);
}

async function optimizeTripV1(
  baseUniverse: number,
  schedule: number,
  tripId: TripId
) {
  const forkResponse = await sendPaxMonForkUniverseRequest({
    universe: baseUniverse,
    fork_schedule: false,
  });
  const simUniverse = forkResponse.universe;
  log(
    `Neues Universum ${simUniverse} für Optimierung erstellt (basierend auf ${baseUniverse}).`
  );
  postMessage({
    type: "UniverseForked",
    universe: simUniverse,
  } as WorkerUpdate);

  try {
    let reload = true;
    let iteration = 1;
    while (reload) {
      log(`Optimierung (V1): Iteration ${iteration}`);
      reload = false;

      const tripLoadData = (
        await sendPaxMonGetTripLoadInfosRequest({
          universe: baseUniverse,
          trips: [tripId],
        })
      ).load_infos[0];
      const groupsInTrip = await sendPaxMonGroupsInTripRequest({
        universe: simUniverse,
        trip: tripId,
        filter: "All",
        group_by_station: "EntryAndLast",
        group_by_other_trip: true,
        include_group_infos: false,
      });

      for (const [sectionIdx, edge] of tripLoadData.edges.entries()) {
        const sectionMeasures = await optimizeTripEdgeV1(
          simUniverse,
          schedule,
          tripId,
          tripLoadData,
          groupsInTrip,
          sectionIdx,
          edge
        );
        if (sectionMeasures.length > 0) {
          log(
            `Optimierung für den Abschnitt: ${sectionMeasures.length} Maßnahmen`
          );
          postMessage({
            type: "MeasuresAdded",
            measures: sectionMeasures,
          } as WorkerUpdate);
          //reload = true; // TODO
          break;
        }
      }
      ++iteration;
    }

    log(`Optimierung abgeschlossen.`);
  } finally {
    log(`Universum ${simUniverse} wird freigegeben...`);
    await sendPaxMonDestroyUniverseRequest({
      universe: simUniverse,
    });
    log(`Universum ${simUniverse} freigegeben.`);
    postMessage({
      type: "UniverseDestroyed",
      universe: simUniverse,
    } as WorkerUpdate);
    postMessage({ type: "OptimizationComplete" } as WorkerUpdate);
  }
}

async function optimizeTripEdgeV1(
  simUniverse: number,
  schedule: number,
  tripId: TripId,
  tripLoadData: PaxMonTripLoadInfo,
  groupsInTrip: PaxMonGetGroupsInTripResponse,
  sectionIdx: number,
  edge: PaxMonEdgeLoadInfo
): Promise<MeasureUnion[]> {
  if (!edge.possibly_over_capacity || edge.prob_over_capacity < 0.01) {
    return [];
  }
  const sectionGroups = groupsInTrip.sections[sectionIdx];
  const minPax = edge.dist.min;
  const maxPax = edge.dist.max;
  let overCap = edge.dist.q95 - edge.capacity;
  log(
    `Kritischer Abschnitt ${sectionIdx}: ${edge.from.name} -> ${edge.to.name}: Kapazität ${edge.capacity}, Reisende: ${minPax}-${maxPax} (Q.95: ${edge.dist.q95}), ${overCap} über Kapazität`
  );

  const optimizedTsi = tripLoadData.tsi;
  const plannedTripId = JSON.stringify(tripId);
  const containsCurrentTrip = (j: Journey) =>
    j.tripLegs.find((leg) =>
      leg.trips.find((t) => JSON.stringify(t.trip.id) === plannedTripId)
    ) !== undefined;

  const sectionMeasures: MeasureUnion[] = [];

  for (const groups of sectionGroups.groups) {
    const destStation = groups.grouped_by_station[0];
    const entryStation = groups.entry_station[0];
    const entryTime = groups.entry_time;
    const previousTrip: TripServiceInfo | undefined = groups.grouped_by_trip[0];

    // TODO: workaround for groups_in_trip api bug
    if (!entryStation) {
      continue;
    }

    const groupSize = groups.info.dist.q50;

    const routingResponse = await sendRoutingRequest({
      start_type: "PretripStart",
      start: {
        station: entryStation,
        interval: {
          begin: entryTime,
          end: entryTime + SEARCH_INTERVAL * 60,
        },
        min_connection_count: 0,
        extend_interval_earlier: false,
        extend_interval_later: false,
      },
      destination: destStation,
      search_type: "Default",
      search_dir: "Forward",
      via: [],
      additional_edges: [],
      use_start_metas: true,
      use_dest_metas: true,
      use_start_footpaths: true,
      schedule,
    });
    const journeys = routingResponse.connections
      .map(connectionToJourney)
      .filter((journey) => journey.tripLegs.length > 0);

    const [currentJourneys, alternativeJourneys] = partition(
      journeys,
      containsCurrentTrip
    );

    if (alternativeJourneys.length === 0) {
      continue;
    }

    const altFirstTrips = alternativeJourneys.map(
      (j) => j.tripLegs[0].trips[0].trip.id
    );
    const altTripLoadInfos = (
      await sendPaxMonGetTripLoadInfosRequest({
        universe: simUniverse,
        trips: altFirstTrips,
      })
    ).load_infos;

    const journeyRating = (j: Journey) =>
      (getArrivalTime(j) - entryTime) / 60 + j.transfers * 30;

    // TODO: trip load infos für alle trip legs abrufen (nicht nur first)
    // TODO: trip load info cache

    const bestCurrentRating = Math.min(...currentJourneys.map(journeyRating));

    const alternatives = zipWith(
      alternativeJourneys,
      altTripLoadInfos,
      (journey, tripLoadInfo) => {
        const rating = journeyRating(journey);
        return {
          journey,
          loadLevel: getLoadLevel(tripLoadInfo, journey.tripLegs[0]),
          rating,
          tsi: tripLoadInfo.tsi,
          estAcceptance: estimateAcceptanceProbability(
            bestCurrentRating,
            rating
          ),
        };
      }
    ).filter((a) => a.loadLevel !== "Full");

    if (alternatives.length === 0) {
      continue;
    }

    alternatives.sort((a, b) => a.rating - b.rating);

    const bestAlternative = alternatives[0];

    const estAcceptance = estimateAcceptanceProbability(
      bestCurrentRating,
      bestAlternative.rating
    );

    if (estAcceptance == 0) {
      continue;
    }

    // TODO: ansagezeitpunkt (trip vs. station)

    const sharedData: SharedMeasureData = {
      time: fromUnixTime(entryTime - 5 * 60),
      recipients: { trips: [], stations: [] },
    };
    if (previousTrip) {
      sharedData.recipients.trips.push(previousTrip);
    } else {
      sharedData.recipients.stations.push(entryStation);
    }

    sectionMeasures.push({
      type: "TripLoadRecommendationMeasure",
      shared: sharedData,
      data: {
        planned_destination: destStation,
        full_trip: { trip: optimizedTsi, level: "Full" },
        recommended_trips: [
          { trip: bestAlternative.tsi, level: bestAlternative.loadLevel },
        ],
      },
    });

    overCap -= groupSize * estAcceptance;

    log(
      `Gruppe: Größe ${groupSize}, ${journeys.length} Alternativen (${
        currentJourneys.length
      } / ${alternativeJourneys.length}), Ratings: ${bestCurrentRating} / ${
        bestAlternative.rating
      } => Accept: ${estAcceptance.toFixed(2)}, Over Cap: ${Math.round(
        overCap
      )}`
    );

    if (overCap <= 0) {
      break;
    }
  }

  return sectionMeasures;
}

async function optimizeTripV2(
  baseUniverse: number,
  schedule: number,
  tripId: TripId
) {
  const forkResponse = await sendPaxMonForkUniverseRequest({
    universe: baseUniverse,
    fork_schedule: false,
  });
  const simUniverse = forkResponse.universe;
  log(
    `Neues Universum ${simUniverse} für Optimierung erstellt (basierend auf ${baseUniverse}).`
  );
  postMessage({
    type: "UniverseForked",
    universe: simUniverse,
  } as WorkerUpdate);

  try {
    let reload = true;
    let iteration = 1;
    while (reload) {
      log(`Optimierung (V2): Iteration ${iteration}`);
      reload = false;

      const tripLoadData = (
        await sendPaxMonGetTripLoadInfosRequest({
          universe: baseUniverse,
          trips: [tripId],
        })
      ).load_infos[0];
      const addressableGroups = await sendPaxMonAddressableGroupsRequest({
        universe: simUniverse,
        trip: tripId,
      });

      const groupMap = new Map<number, PaxMonGroupBaseInfo>();
      for (const group of addressableGroups.groups) {
        groupMap.set(group.id, group);
      }

      for (const [sectionIdx, edge] of tripLoadData.edges.entries()) {
        const sectionMeasures = await optimizeTripEdgeV2(
          simUniverse,
          schedule,
          tripId,
          tripLoadData,
          addressableGroups,
          groupMap,
          sectionIdx,
          edge
        );
        if (sectionMeasures.length > 0) {
          log(
            `Optimierung für den Abschnitt: ${sectionMeasures.length} Maßnahmen`
          );
          postMessage({
            type: "MeasuresAdded",
            measures: sectionMeasures,
          } as WorkerUpdate);
          //reload = true; // TODO
          break;
        }
      }
      ++iteration;
    }

    log(`Optimierung abgeschlossen.`);
  } finally {
    log(`Universum ${simUniverse} wird freigegeben...`);
    await sendPaxMonDestroyUniverseRequest({
      universe: simUniverse,
    });
    log(`Universum ${simUniverse} freigegeben.`);
    postMessage({
      type: "UniverseDestroyed",
      universe: simUniverse,
    } as WorkerUpdate);
  }
}

function estimateAcceptanceProbability(
  bestCurrentRating: number,
  bestAlternativeRating: number
): number {
  if (bestAlternativeRating === Infinity) {
    return 0;
  } else if (bestCurrentRating === Infinity) {
    return 1;
  } else if (bestAlternativeRating < bestCurrentRating) {
    return 1;
  } else {
    return (
      1.0 - Math.min(0.8, (bestAlternativeRating - bestCurrentRating) / 90)
    ); // TODO
  }
}

async function optimizeTripEdgeV2(
  simUniverse: number,
  schedule: number,
  tripId: TripId,
  tripLoadData: PaxMonTripLoadInfo,
  addressableGroups: PaxMonGetAddressableGroupsResponse,
  groupMap: Map<number, PaxMonGroupBaseInfo>,
  sectionIdx: number,
  edge: PaxMonEdgeLoadInfo
): Promise<MeasureUnion[]> {
  if (!edge.possibly_over_capacity || edge.prob_over_capacity < 0.01) {
    return [];
  }
  const sectionGroupInfo = addressableGroups.sections[sectionIdx];
  const minPax = edge.dist.min;
  const maxPax = edge.dist.max;
  const overCap = edge.dist.q95 - edge.capacity;
  log(
    `\n\nKritischer Abschnitt ${sectionIdx}: ${edge.from.name} -> ${edge.to.name}: Kapazität ${edge.capacity}, Reisende: ${minPax}-${maxPax} (Q.95: ${edge.dist.q95}), ${overCap} über Kapazität`
  );

  /*
  const optimizedTsi = tripLoadData.tsi;
  const plannedTripId = JSON.stringify(tripId);
  const containsCurrentTrip = (j: Journey) =>
    j.tripLegs.find((leg) =>
      leg.trips.find((t) => JSON.stringify(t.trip.id) === plannedTripId)
    ) !== undefined;
   */

  const sectionMeasures: MeasureUnion[] = [];

  for (const byInterchange of sectionGroupInfo.by_future_interchange) {
    const destStation = byInterchange.future_interchange;
    const cgs = byInterchange.cgs;

    log(
      `\nReisende Richtung ${destStation.name}: ~${cgs.dist.q5}-${cgs.dist.q95}`
    );
    for (const byEntry of byInterchange.by_entry) {
      log(
        `  Einstieg in ${byEntry.entry_station.name}: ~${byEntry.cgs.dist.q5}-${byEntry.cgs.dist.q95}`
      );
      log(
        `    Reisebeginn an der Station: ~${byEntry.starting_here.dist.q5}-${byEntry.starting_here.dist.q95}`
      );
      for (const byFeeder of byEntry.by_feeder) {
        log(
          `    Umstieg von Zug #${byFeeder.trip.trip.train_nr}: ~${byFeeder.cgs.dist.q5}-${byFeeder.cgs.dist.q95}`
        );
      }
    }
  }

  return sectionMeasures;
}

const LOAD_LEVEL_MAPPING: LoadLevel[] = ["Low", "NoSeats", "Full"];

function getLoadLevel(
  tripLoadInfo: PaxMonTripLoadInfo,
  tripLeg: TripLeg
): LoadLevel {
  let level = 0;
  let inTrip = false;

  const entryStop = tripLeg.stops[0];
  const exitStop = tripLeg.stops[tripLeg.stops.length - 1];

  for (const edge of tripLoadInfo.edges) {
    // TODO: check times
    if (!inTrip && edge.from.id === entryStop.station.id) {
      inTrip = true;
    }
    if (inTrip) {
      if (edge.capacity_type === "Known") {
        const expectedLoad = edge.dist.q95 / edge.capacity;
        if (expectedLoad > 1.0) {
          level = 2; // Full
        } else if (expectedLoad > 0.8) {
          level = Math.max(level, 1); // NoSeats
        }
      }

      if (edge.to.id === exitStop.station.id) {
        inTrip = false;
        break;
      }
    }
  }
  return LOAD_LEVEL_MAPPING[level];
}

async function optimizeTrip(
  baseUniverse: number,
  schedule: number,
  tripId: TripId,
  optType: OptimizationType
) {
  switch (optType) {
    case "LoadInfo":
      return optimizeTripV1(baseUniverse, schedule, tripId);
    case "LoadRecommendation":
      return optimizeTripV2(baseUniverse, schedule, tripId);
  }
}

onmessage = (msg) => {
  const req = msg.data as WorkerRequest;

  switch (req.action) {
    case "Init": {
      setApiEndpoint(req.apiEndpoint);
      break;
    }
    case "Start": {
      optimizeTrip(req.universe, req.schedule, req.tripId, req.optType);
      break;
    }
  }
};
