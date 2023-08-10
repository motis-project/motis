import { PaxMonCapacitySource } from "@/api/protocol/motis/paxmon";

export function getCapacitySourceTooltip(cs: PaxMonCapacitySource): string {
  switch (cs) {
    case "FormationVehicles":
      return "Kapazitätsinformationen aus Wagenreihung und Fahrzeugnummern";
    case "FormationVehicleGroups":
      return "Kapazitätsinformationen aus Wagenreihung und Fahrzeuggruppen";
    case "FormationBaureihe":
      return "Kapazitätsinformationen aus Wagenreihung und Baureihen";
    case "FormationGattung":
      return "Kapazitätsinformationen aus Wagenreihung und Fahrzeuggattungen";
    case "TripExactMatch":
      return "Kapazitätsinformationen für den Zug gefunden";
    case "TripPrimaryIdMatch":
      return "Kapazitätsinformationen für den Zug gefunden (nur Übereinstimmung der primären Trip Id)";
    case "TrainNrAndStations":
      return "Kapazitätsinformationen möglicherweise falsch - nur Übereinstimmung der Zugnummer und Start-/Zielstationen";
    case "TrainNr":
      return "Kapazitätsinformationen möglicherweise falsch - nur Übereinstimmung der Zugnummer";
    case "Category":
      return "Keine zugspezifischen Kapazitätsinformationen vorhanden, Standardwert für Zugkategorie";
    case "Class":
      return "Keine zugspezifischen Kapazitätsinformationen vorhanden, Standardwert für Zugklasse";
    case "Override":
      return "Kapazitätsinformationen durch Simulation manuell überschrieben";
    case "Unlimited":
      return "Unbegrenzte Kapazität";
    case "Unknown":
      return "Keine Kapazitätsinformationen vorhanden";
  }
}

export function getCapacitySourceShortText(cs: PaxMonCapacitySource): string {
  switch (cs) {
    case "FormationVehicles":
      return "Wagenreihung";
    case "FormationVehicleGroups":
      return "WR + Gruppe";
    case "FormationBaureihe":
      return "WR + Baureihe";
    case "FormationGattung":
      return "WR + Gattung";
    case "TripExactMatch":
      return "Exakt";
    case "TripPrimaryIdMatch":
      return "Primary TID";
    case "TrainNrAndStations":
      return "ZugNr + St.";
    case "TrainNr":
      return "ZugNr";
    case "Category":
      return "Kategorie";
    case "Class":
      return "Klasse";
    case "Override":
      return "Manuell";
    case "Unlimited":
      return "Unbegrenzt";
    case "Unknown":
      return "Unbekannt";
  }
}

export function getFormationCapacitySourceShortText(
  cs: PaxMonCapacitySource,
  single: boolean,
): string {
  switch (cs) {
    case "FormationVehicles":
      return single ? "Fahrzeugnummer" : "Fahrzeugnummern";
    case "FormationVehicleGroups":
      return "Fahrzeuggruppe";
    case "FormationBaureihe":
      return "Baureihe";
    case "FormationGattung":
      return "Fahrzeuggattung";
    case "Unknown":
      return "Unbekannt";
    default:
      return "???";
  }
}

const EXACT_SOURCES: PaxMonCapacitySource[] = [
  "FormationVehicles",
  "FormationVehicleGroups",
  "TripExactMatch",
];

export function isExactCapacitySource(cs: PaxMonCapacitySource): boolean {
  return EXACT_SOURCES.includes(cs);
}
