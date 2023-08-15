import { v4 as uuidv4 } from "uuid";

import { Station } from "@/api/protocol/motis";
import {
  RiBasisFahrtData,
  RiBasisGattung,
  RiBasisHaltestelle,
  RiBasisLinie,
  RiBasisVerwaltung,
} from "@/api/protocol/motis/ribasis";

export function getRiBasisGattung(
  data: RiBasisFahrtData,
  id: string,
): RiBasisGattung | undefined {
  for (const g of data.allGattung) {
    if (g.gattungid === id) {
      return g;
    }
  }
  return undefined;
}

export function getOrAddRiBasisGattung(
  data: RiBasisFahrtData,
  name: string,
): RiBasisGattung {
  for (const g of data.allGattung) {
    if (g.name === name) {
      return g;
    }
  }
  const g = { gattungid: uuidv4(), name, code: name };
  data.allGattung.push(g);
  return g;
}

export function getRiBasisLinie(
  data: RiBasisFahrtData,
  id: string,
): RiBasisLinie | undefined {
  for (const l of data.allLinie) {
    if (l.linieid === id) {
      return l;
    }
  }
  return undefined;
}

export function getOrAddRiBasisLinie(
  data: RiBasisFahrtData,
  name: string,
): RiBasisLinie {
  for (const l of data.allLinie) {
    if (l.name === name) {
      return l;
    }
  }
  const l = { linieid: uuidv4(), name };
  data.allLinie.push(l);
  return l;
}

export function getRiBasisVerwaltung(
  data: RiBasisFahrtData,
  id: string,
): RiBasisVerwaltung | undefined {
  for (const v of data.allVerwaltung) {
    if (v.verwaltungid === id) {
      return v;
    }
  }
  return undefined;
}

export function getOrAddRiBasisVerwaltung(
  data: RiBasisFahrtData,
  name: string,
): RiBasisVerwaltung {
  for (const v of data.allVerwaltung) {
    if (v.betreiber.name === name) {
      return v;
    }
  }
  const v = { verwaltungid: uuidv4(), betreiber: { name, code: name } };
  data.allVerwaltung.push(v);
  return v;
}

export function riBasisHaltestelleToMotisStation(
  s: RiBasisHaltestelle,
): Station {
  return { id: s.evanummer, name: s.bezeichnung, pos: { lat: 0, lng: 0 } };
}

export function motisStationToRiBasisHaltestelle(
  s: Station,
): RiBasisHaltestelle {
  return {
    haltestelleid: uuidv4(),
    bezeichnung: s.name,
    evanummer: s.id,
    rl100: "",
  };
}
