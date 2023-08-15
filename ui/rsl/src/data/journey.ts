import {
  Connection,
  Range,
  Stop,
  Transport,
  Trip,
  Walk,
} from "@/api/protocol/motis";

export interface JourneyTransport {
  transport: Transport;
  trips: Trip[];
}

export interface JourneyTrip {
  trip: Trip;
  transports: Transport[];
}

export interface TripLeg {
  type: "trip";
  stops: Stop[];
  transports: JourneyTransport[];
  trips: JourneyTrip[];
}

export interface WalkLeg {
  type: "walk";
  from: Stop;
  to: Stop;
  walk: Walk;
}

export type JourneyLeg = TripLeg | WalkLeg;

export interface Journey {
  legs: JourneyLeg[];
  tripLegs: TripLeg[];
  walkLegs: WalkLeg[];
  exits: number;
  transfers: number;
  con: Connection;
}

interface Ranged {
  range: Range;
}

function getWalk(con: Connection, fromIdx: number, toIdx: number): Walk {
  for (const mw of con.transports) {
    if (mw.move_type == "Walk") {
      const w = mw.move as Walk;
      if (w.range.from == fromIdx && w.range.to == toIdx) {
        return w;
      }
    }
  }
  throw new Error("walk transport not found");
}

function overlapsRange(range: Range, fromIdx: number, toIdx: number): boolean {
  return range.from < toIdx && range.to > fromIdx;
}

function rangeSort<T extends Ranged>(a: T, b: T): number {
  return b.range.to - b.range.from - (a.range.to - a.range.from);
}

function getTrips(con: Connection, fromIdx: number, toIdx: number): Trip[] {
  return con.trips
    .filter((t) => overlapsRange(t.range, fromIdx, toIdx))
    .sort(rangeSort);
}

function getTransports(
  con: Connection,
  fromIdx: number,
  toIdx: number,
): Transport[] {
  return con.transports
    .filter((mw) => {
      if (mw.move_type == "Transport") {
        const t = mw.move as Transport;
        return overlapsRange(t.range, fromIdx, toIdx);
      }
      return false;
    })
    .map((mw) => mw.move as Transport)
    .sort(rangeSort);
}

function getJourneyTransports(
  con: Connection,
  fromIdx: number,
  toIdx: number,
): JourneyTransport[] {
  const res: JourneyTransport[] = [];
  for (const mw of con.transports) {
    if (mw.move_type == "Transport") {
      const transport = mw.move as Transport;
      if (overlapsRange(transport.range, fromIdx, toIdx)) {
        res.push({
          transport: transport,
          trips: getTrips(con, transport.range.from, transport.range.to),
        });
      }
    }
  }
  return res;
}

function getJourneyTrips(
  con: Connection,
  fromIdx: number,
  toIdx: number,
): JourneyTrip[] {
  return con.trips
    .filter((t) => overlapsRange(t.range, fromIdx, toIdx))
    .map((t) => {
      return {
        trip: t,
        transports: getTransports(con, t.range.from, t.range.to),
      };
    });
}

export function connectionToJourney(con: Connection): Journey {
  const j: Journey = {
    legs: [],
    tripLegs: [],
    walkLegs: [],
    exits: 0,
    transfers: 0,
    con,
  };

  let currentLeg: TripLeg | null = null;
  let enterIdx = -1;
  for (let i = 0; i < con.stops.length; ++i) {
    const stop = con.stops[i];
    if (currentLeg) {
      currentLeg.stops.push(stop);
    }
    if (stop.exit) {
      if (currentLeg) {
        currentLeg.transports = getJourneyTransports(con, enterIdx, i);
        currentLeg.trips = getJourneyTrips(con, enterIdx, i);
        j.legs.push(currentLeg);
        j.exits++;
        currentLeg = null;
        enterIdx = -1;
      } else {
        throw new Error("invalid connection (unexpected exit)");
      }
    }
    if (stop.enter) {
      currentLeg = { type: "trip", stops: [stop], transports: [], trips: [] };
      enterIdx = i;
    }
    if (!currentLeg && i !== con.stops.length - 1) {
      j.legs.push({
        type: "walk",
        from: stop,
        to: con.stops[i + 1],
        walk: getWalk(con, i, i + 1),
      });
    }
  }

  j.transfers = Math.max(0, j.exits - 1);
  j.tripLegs = j.legs.filter((l) => l.type === "trip") as TripLeg[];
  j.walkLegs = j.legs.filter((l) => l.type === "walk") as WalkLeg[];

  return j;
}

export function getArrivalTime(j: Journey): number {
  const finalLeg = j.legs[j.legs.length - 1];
  switch (finalLeg.type) {
    case "trip":
      return finalLeg.stops[finalLeg.stops.length - 1].arrival.time;
    case "walk":
      return finalLeg.to.arrival.time;
  }
}

export function getDepartureTime(j: Journey): number {
  const firstLeg = j.legs[0];
  switch (firstLeg.type) {
    case "trip":
      return firstLeg.stops[0].departure.time;
    case "walk":
      return firstLeg.from.departure.time;
  }
}
