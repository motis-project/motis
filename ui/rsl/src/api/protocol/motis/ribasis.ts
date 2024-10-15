// generated file - do not modify - run update-protocol to update

// ribasis/RiBasisFahrt.fbs
export interface RiBasisMeta {
  id: string;
  owner: string;
  format: string;
  version: string;
  correlation: string[];
  created: string;
  sequence: number;
}

// ribasis/RiBasisFahrt.fbs
export type RiBasisFahrtKategorie = "SOLL" | "IST" | "VORSCHAU";

// ribasis/RiBasisFahrt.fbs
export type RiBasisFahrtTyp =
  | "PLANFAHRT"
  | "ERSATZFAHRT"
  | "ENTLASTUNGSFAHRT"
  | "SONDERFAHRT";

// ribasis/RiBasisFahrt.fbs
export interface RiBasisHaltestelle {
  haltestelleid: string;
  bezeichnung: string;
  evanummer: string;
  rl100: string;
}

// ribasis/RiBasisFahrt.fbs
export interface RiBasisBetreiber {
  name: string;
  code: string;
}

// ribasis/RiBasisFahrt.fbs
export interface RiBasisVerwaltung {
  verwaltungid: string;
  betreiber: RiBasisBetreiber;
}

// ribasis/RiBasisFahrt.fbs
export interface RiBasisGattung {
  gattungid: string;
  name: string;
  code: string;
}

// ribasis/RiBasisFahrt.fbs
export interface RiBasisLinie {
  linieid: string;
  name: string;
}

// ribasis/RiBasisFahrt.fbs
export interface RiBasisFahrtRelation {
  startfahrtnummer: string;
  startzeit: string;
  startverwaltung: string;
  startgattung: string;
  startlinie: string;
  starthaltestelle: RiBasisHaltestelle;
  zielzeit: string;
  zielhaltestelle: RiBasisHaltestelle;
}

// ribasis/RiBasisFahrt.fbs
export type RiBasisZeitstatus =
  | "FAHRPLAN"
  | "MELDUNG"
  | "AUTOMAT"
  | "PROGNOSE"
  | "UNBEKANNT";

// ribasis/RiBasisFahrt.fbs
export type RiBasisOrtTyp = "STEIG" | "GLEIS";

// ribasis/RiBasisFahrt.fbs
export interface RiBasisOrt {
  ortid: string;
  bezeichnung: string;
  orttyp: RiBasisOrtTyp;
}

// ribasis/RiBasisFahrt.fbs
export type RiBasisHaltzuordnungstyp =
  | "IST_ERSATZ_FUER"
  | "WIRD_ERSETZT_DURCH"
  | "IST_ENTLASTUNG_FUER"
  | "WIRD_ENTLASTET_DURCH"
  | "GLEISAENDERUNG_VON"
  | "GLEISAENDERUNG_NACH";

// ribasis/RiBasisFahrt.fbs
export interface RiBasisAbfahrtZuordnung {
  fahrtid: string;
  abfahrtid: string;
  typ: RiBasisHaltzuordnungstyp;
}

// ribasis/RiBasisFahrt.fbs
export interface RiBasisAbfahrt {
  abfahrtid: string;
  haltestelle: RiBasisHaltestelle;
  fahrgastwechsel: boolean;
  planabfahrtzeit: string;
  abfahrtzeit: string;
  abfahrtzeitstatus: RiBasisZeitstatus;
  planabfahrtort: RiBasisOrt;
  abfahrtort: RiBasisOrt;
  zusatzhalt: boolean;
  bedarfshalt: boolean;
  allAbfahrtzuordnung: RiBasisAbfahrtZuordnung[];
}

// ribasis/RiBasisFahrt.fbs
export interface RiBasisAnkunftZuordnung {
  fahrtid: string;
  ankunftid: string;
  typ: RiBasisHaltzuordnungstyp;
}

// ribasis/RiBasisFahrt.fbs
export interface RiBasisAnkunft {
  ankunftid: string;
  haltestelle: RiBasisHaltestelle;
  fahrgastwechsel: boolean;
  planankunftzeit: string;
  ankunftzeit: string;
  ankunftzeitstatus: RiBasisZeitstatus;
  planankunftort: RiBasisOrt;
  ankunftort: RiBasisOrt;
  zusatzhalt: boolean;
  bedarfshalt: boolean;
  allAnkunftzuordnung: RiBasisAnkunftZuordnung[];
}

// ribasis/RiBasisFahrt.fbs
export interface RiBasisFormation {
  fahrtid: string;
  abfahrtid: string;
  ankunftid: string;
}

// ribasis/RiBasisFahrt.fbs
export interface RiBasisFahrtAbschnitt {
  fahrtnummer: string;
  fahrtbezeichnung: string;
  fahrtname: string;
  verwaltungid: string;
  gattungid: string;
  linieid: string;
  abfahrt: RiBasisAbfahrt;
  ankunft: RiBasisAnkunft;
  allVereinigtmit: RiBasisFormation[];
}

// ribasis/RiBasisFahrt.fbs
export type RiBasisFahrtZuordnungstyp = "DURCHBINDUNG" | "WENDE";

// ribasis/RiBasisFahrt.fbs
export interface RiBasisZubringerFahrtZuordnung {
  fahrtid: string;
  ankunftid: string;
  typ: RiBasisFahrtZuordnungstyp;
}

// ribasis/RiBasisFahrt.fbs
export interface RiBasisAbbringerFahrtZuordnung {
  fahrtid: string;
  abfahrtid: string;
  typ: RiBasisFahrtZuordnungstyp;
}

// ribasis/RiBasisFahrt.fbs
export interface RiBasisFahrtData {
  kategorie: RiBasisFahrtKategorie;
  planstarttag: string;
  fahrtid: string;
  fahrtrelation: RiBasisFahrtRelation;
  verkehrstag: string;
  fahrttyp: RiBasisFahrtTyp;
  allVerwaltung: RiBasisVerwaltung[];
  allGattung: RiBasisGattung[];
  allLinie: RiBasisLinie[];
  allFahrtabschnitt: RiBasisFahrtAbschnitt[];
  allZubringerfahrtzuordnung: RiBasisZubringerFahrtZuordnung[];
  allAbbringerfahrtzuordnung: RiBasisAbbringerFahrtZuordnung[];
}

// ribasis/RiBasisFahrt.fbs
export interface RiBasisFahrt {
  meta: RiBasisMeta;
  data: RiBasisFahrtData;
}
