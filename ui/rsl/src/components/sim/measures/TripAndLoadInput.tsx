import React, { ReactNode } from "react";

import { TripServiceInfo } from "@/api/protocol/motis";
import { LoadLevel } from "@/api/protocol/motis/paxforecast";

import LoadInput, { allLoadLevels } from "@/components/inputs/LoadInput";
import TripPicker from "@/components/inputs/TripPicker";

export interface TripAndLoadInputProps {
  selectedTrip: TripServiceInfo | undefined;
  selectedLevel: LoadLevel;
  onTripSelected: (tsi: TripServiceInfo | undefined) => void;
  onLevelSelected: (level: LoadLevel) => void;
  loadLevels?: LoadLevel[];
  children?: ReactNode;
}

function TripAndLoadInput({
  selectedTrip,
  selectedLevel,
  onTripSelected,
  onLevelSelected,
  loadLevels = allLoadLevels,
  children,
}: TripAndLoadInputProps) {
  return (
    <div className="flex justify-between items-center gap-2">
      <TripPicker
        onTripPicked={onTripSelected}
        clearOnPick={false}
        longDistanceOnly={false}
        initialTrip={selectedTrip}
        key={JSON.stringify(selectedTrip)}
        className="w-32 flex-shrink-0"
      />
      <LoadInput
        loadLevels={loadLevels}
        selectedLevel={selectedLevel}
        onLevelSelected={onLevelSelected}
        className="flex-grow"
      />
      {children}
    </div>
  );
}

export default TripAndLoadInput;
