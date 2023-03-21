import { PaxMonCapacitySource } from "@/api/protocol/motis/paxmon";

export function getCapacitySourceTooltip(cs: PaxMonCapacitySource): string {
  switch (cs) {
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
    case "Unknown":
      return "Keine Kapazitätsinformationen vorhanden";
  }
}

export function getCapacitySourceShortText(cs: PaxMonCapacitySource): string {
  switch (cs) {
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
    case "Unknown":
      return "Unbekannt";
  }
}
