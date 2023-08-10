import { addMinutes, differenceInMinutes, subMinutes } from "date-fns";
import { v4 as uuidv4 } from "uuid";

import { Station } from "@/api/protocol/motis";
import {
  RiBasisFahrtAbschnitt,
  RiBasisFahrtData,
  RiBasisHaltestelle,
  RiBasisOrt,
  RiBasisZeitstatus,
} from "@/api/protocol/motis/ribasis";

import { formatRiBasisDateTime, parseRiBasisDateTime } from "@/util/dateFormat";
import {
  getOrAddRiBasisGattung,
  getOrAddRiBasisLinie,
  getOrAddRiBasisVerwaltung,
  getRiBasisGattung,
  getRiBasisLinie,
  motisStationToRiBasisHaltestelle,
  riBasisHaltestelleToMotisStation,
} from "@/util/ribasis";

export interface StopFormData {
  station: Station;
  arrival: EventFormData; // 1st stop: copy of departure, ignored
  departure: EventFormData; // last stop: copy of arrival, ignored
  section: SectionFormData; // last stop: copy of previous section, ignored
}

export interface EventFormData {
  scheduleTime: Date;
  currentTime: Date;
  reason: RiBasisZeitstatus;
  scheduleTrack: string;
  currentTrack: string;
  interchange: boolean;
}

export interface SectionFormData {
  trainNr: number;
  category: string;
  line: string;
}

function getArrivalEvent(
  ribasis: RiBasisFahrtData,
  stopIdx: number,
): EventFormData {
  if (stopIdx > 0) {
    const sec = ribasis.allFahrtabschnitt[stopIdx - 1];
    const arr = sec.ankunft;
    return {
      scheduleTime: parseRiBasisDateTime(arr.planankunftzeit),
      currentTime: parseRiBasisDateTime(arr.ankunftzeit),
      reason: arr.ankunftzeitstatus,
      scheduleTrack: arr.planankunftort.bezeichnung,
      currentTrack: arr.ankunftort.bezeichnung,
      interchange: arr.fahrgastwechsel,
    };
  } else {
    const arr = getDepartureEvent(ribasis, stopIdx);
    arr.scheduleTime = subMinutes(arr.scheduleTime, 2);
    arr.currentTime = subMinutes(arr.currentTime, 2);
    return arr;
  }
}

function getDepartureEvent(
  ribasis: RiBasisFahrtData,
  stopIdx: number,
): EventFormData {
  if (stopIdx < ribasis.allFahrtabschnitt.length) {
    const sec = ribasis.allFahrtabschnitt[stopIdx];
    const dep = sec.abfahrt;
    return {
      scheduleTime: parseRiBasisDateTime(dep.planabfahrtzeit),
      currentTime: parseRiBasisDateTime(dep.abfahrtzeit),
      reason: dep.abfahrtzeitstatus,
      scheduleTrack: dep.planabfahrtort.bezeichnung,
      currentTrack: dep.abfahrtort.bezeichnung,
      interchange: dep.fahrgastwechsel,
    };
  } else {
    const dep = getArrivalEvent(ribasis, stopIdx);
    dep.scheduleTime = addMinutes(dep.scheduleTime, 2);
    dep.currentTime = addMinutes(dep.currentTime, 2);
    return dep;
  }
}

function getStation(
  ribasis: RiBasisFahrtData,
  stopIdx: number,
): RiBasisHaltestelle {
  if (stopIdx > 0) {
    return ribasis.allFahrtabschnitt[stopIdx - 1].ankunft.haltestelle;
  } else {
    return ribasis.allFahrtabschnitt[stopIdx].abfahrt.haltestelle;
  }
}

function getSection(
  ribasis: RiBasisFahrtData,
  stopIdx: number,
): RiBasisFahrtAbschnitt {
  if (stopIdx < ribasis.allFahrtabschnitt.length) {
    return ribasis.allFahrtabschnitt[stopIdx];
  } else {
    return ribasis.allFahrtabschnitt[stopIdx - 1];
  }
}

export function toFormData(ribasis: RiBasisFahrtData): StopFormData[] {
  const stops: StopFormData[] = [];
  const sectionCount = ribasis.allFahrtabschnitt.length;
  if (sectionCount == 0) {
    return stops;
  }
  for (let stopIdx = 0; stopIdx < sectionCount + 1; stopIdx++) {
    const depSec = getSection(ribasis, stopIdx);
    stops.push({
      station: riBasisHaltestelleToMotisStation(getStation(ribasis, stopIdx)),
      arrival: getArrivalEvent(ribasis, stopIdx),
      departure: getDepartureEvent(ribasis, stopIdx),
      section: {
        trainNr: Number.parseInt(depSec.fahrtnummer),
        category: getRiBasisGattung(ribasis, depSec.gattungid)?.name ?? "",
        line: getRiBasisLinie(ribasis, depSec.linieid)?.name ?? "",
      },
    });
  }

  return stops;
}

export function getEmptyEventFormData(time: Date): EventFormData {
  return {
    currentTime: time,
    currentTrack: "",
    interchange: true,
    reason: "FAHRPLAN",
    scheduleTime: time,
    scheduleTrack: "",
  };
}

export function getEmptyStation(): Station {
  return { name: "", id: "", pos: { lat: 0, lng: 0 } };
}

export function getEmptySectionFormData(): SectionFormData {
  return { trainNr: 0, category: "", line: "" };
}

export function getEmptyStopFormData(time: Date): StopFormData {
  return {
    station: getEmptyStation(),
    arrival: getEmptyEventFormData(time),
    departure: getEmptyEventFormData(time),
    section: getEmptySectionFormData(),
  };
}

export function getNewEventFormData(
  scheduleTime: Date,
  currentTime: Date,
): EventFormData {
  return {
    scheduleTime,
    currentTime,
    reason: scheduleTime == currentTime ? "FAHRPLAN" : "PROGNOSE",
    scheduleTrack: "",
    currentTrack: "",
    interchange: true,
  };
}

export function getNewStopTimes(
  previousStop: StopFormData | undefined,
  nextStop: StopFormData | undefined,
  fallbackTime: Date,
): {
  currentArrival: Date;
  scheduleArrival: Date;
  currentDeparture: Date;
  scheduleDeparture: Date;
} {
  let scheduleArrival = fallbackTime;
  let scheduleDeparture = fallbackTime;
  let currentArrival = fallbackTime;
  let currentDeparture = fallbackTime;

  if (previousStop && nextStop) {
    const scheduleDiff = differenceInMinutes(
      nextStop.arrival.scheduleTime,
      previousStop.departure.scheduleTime,
    );
    const currentDiff = differenceInMinutes(
      nextStop.arrival.currentTime,
      previousStop.departure.currentTime,
    );
    const scheduleMid = addMinutes(
      previousStop.departure.scheduleTime,
      scheduleDiff / 2,
    );
    const currentMid = addMinutes(
      previousStop.departure.currentTime,
      currentDiff / 2,
    );
    scheduleArrival =
      scheduleDiff > 4 ? subMinutes(scheduleMid, 1) : scheduleMid;
    scheduleDeparture =
      scheduleDiff > 4 ? addMinutes(scheduleMid, 1) : scheduleMid;
    currentArrival = currentDiff > 4 ? subMinutes(currentMid, 1) : currentMid;
    currentDeparture = currentDiff > 4 ? addMinutes(currentMid, 1) : currentMid;
  } else if (previousStop) {
    scheduleArrival = addMinutes(previousStop.departure.scheduleTime, 10);
    currentArrival = addMinutes(previousStop.departure.currentTime, 10);
    scheduleDeparture = scheduleArrival;
    currentDeparture = currentArrival;
  } else if (nextStop) {
    scheduleDeparture = subMinutes(nextStop.arrival.scheduleTime, 10);
    currentDeparture = subMinutes(nextStop.arrival.currentTime, 10);
    scheduleArrival = scheduleDeparture;
    currentArrival = currentDeparture;
  }

  return {
    scheduleArrival,
    scheduleDeparture,
    currentArrival,
    currentDeparture,
  };
}

export function toRiBasis(
  original: RiBasisFahrtData,
  stops: StopFormData[],
): RiBasisFahrtData {
  const data: RiBasisFahrtData = {
    ...original,
    allVerwaltung: [...original.allVerwaltung],
    allGattung: [...original.allGattung],
    allLinie: [...original.allLinie],
    allFahrtabschnitt: [],
    allZubringerfahrtzuordnung: [],
    allAbbringerfahrtzuordnung: [],
  };

  const trackMap = new Map<string, RiBasisOrt>();
  const toTrack = (track: string): RiBasisOrt => {
    let t = trackMap.get(track);
    if (!t) {
      t = { ortid: uuidv4(), bezeichnung: track, orttyp: "GLEIS" };
      trackMap.set(track, t);
    }
    return t;
  };

  const sectionCount = stops.length - 1;
  for (let sectionIdx = 0; sectionIdx < sectionCount; sectionIdx++) {
    const depStop = stops[sectionIdx];
    const arrStop = stops[sectionIdx + 1];
    const sec = depStop.section;
    const dep = depStop.departure;
    const arr = arrStop.arrival;
    if (!sec || !dep || !arr) {
      throw new Error("toRiBasis: invalid section");
    }
    data.allFahrtabschnitt.push({
      fahrtnummer: sec.trainNr.toString(),
      fahrtbezeichnung: `${sec.category} ${sec.trainNr}`,
      fahrtname: "",
      verwaltungid: getOrAddRiBasisVerwaltung(data, "").verwaltungid,
      gattungid: getOrAddRiBasisGattung(data, sec.category).gattungid,
      linieid: getOrAddRiBasisLinie(data, sec.line).linieid,
      abfahrt: {
        abfahrtid: uuidv4(),
        haltestelle: motisStationToRiBasisHaltestelle(depStop.station),
        fahrgastwechsel: dep.interchange,
        planabfahrtzeit: formatRiBasisDateTime(dep.scheduleTime),
        abfahrtzeit: formatRiBasisDateTime(dep.currentTime),
        abfahrtzeitstatus: dep.reason,
        planabfahrtort: toTrack(dep.scheduleTrack),
        abfahrtort: toTrack(dep.currentTrack),
        zusatzhalt: false,
        bedarfshalt: false,
        allAbfahrtzuordnung: [],
      },
      ankunft: {
        ankunftid: uuidv4(),
        haltestelle: motisStationToRiBasisHaltestelle(arrStop.station),
        fahrgastwechsel: arr.interchange,
        planankunftzeit: formatRiBasisDateTime(arr.scheduleTime),
        ankunftzeit: formatRiBasisDateTime(arr.currentTime),
        ankunftzeitstatus: arr.reason,
        planankunftort: toTrack(arr.scheduleTrack),
        ankunftort: toTrack(arr.currentTrack),
        zusatzhalt: false,
        bedarfshalt: false,
        allAnkunftzuordnung: [],
      },
      allVereinigtmit: [],
    });
  }
  return data;
}
