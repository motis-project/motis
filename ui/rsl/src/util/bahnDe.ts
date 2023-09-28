import { format } from "date-fns";

import { Station, TripId } from "@/api/protocol/motis";

import { getDate } from "@/util/dateFormat";

export function guessBahnOrtId(station: Station): string {
  const ts = Math.round(Date.now());
  const x = Math.round(station.pos.lng * 1e6);
  const y = Math.round(station.pos.lat * 1e6);
  return `A=1@O=${station.name}@X=${x}@Y=${y}@U=80@L=${station.id}@B=1@p=${ts}@`;
}

export function getBahnSucheUrl(
  from: Station,
  to: Station,
  departure: Date | number,
): string {
  const params = new URLSearchParams({
    sts: "true",
    so: from.name, // Startort
    zo: to.name, // Zielort
    kl: "2", // Klasse
    r: "13:16:KLASSENLOS:1", // Reisende (1 Erwachsener)
    soid: guessBahnOrtId(from),
    zoid: guessBahnOrtId(to),
    hd: format(getDate(departure), "yyyy-MM-dd'T'HH:mm:ss"),
    hza: "D", // Abfahrtszeit
    ar: "false", // Autonome Reservierung
    s: "false", // Schnellste Verbindungen
    d: "false", // Nur Direktverbindungen
    hz: "[]", // Hinfahrt Zwischenhalte
    fm: "false", // Fahrradmitnahme
    bp: "false", // Bestpreissuche
  });
  return "https://www.bahn.de/buchung/fahrplan/suche#" + params.toString();
}

export function getBahnTrainSearchUrl(
  trip: TripId,
  station?: Station | undefined | null,
): string {
  const params = new URLSearchParams({
    ld: "4336",
    country: "DEU",
    protocol: "https:",
    rt: "1",
    trainname: trip.train_nr.toString(),
    date: format(getDate(trip.time), "dd.MM.yy"),
  });
  if (station) {
    params.set("stationname", station.name);
    params.set("REQ0JourneyStopsSID", guessBahnOrtId(station));
  }
  return (
    "https://reiseauskunft.bahn.de/bin/trainsearch.exe/dn?" + params.toString()
  );
}
