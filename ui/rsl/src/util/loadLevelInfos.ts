import { LoadLevel } from "@/api/protocol/motis/paxforecast";

export interface LoadLevelInfo {
  label: string;
  bgColor: string;
}

export const loadLevelInfos: Record<LoadLevel, LoadLevelInfo> = {
  Unknown: {
    label: "Unbekannte Auslastung",
    bgColor: "bg-db-cool-gray-300",
  },
  Low: { label: "Geringe Auslastung", bgColor: "bg-green-500" },
  NoSeats: {
    label: "Keine Sitzplätze",
    bgColor: "bg-yellow-500",
  },
  Full: {
    label: "Keine Mitfahrmöglichkeit",
    bgColor: "bg-red-500",
  },
};
