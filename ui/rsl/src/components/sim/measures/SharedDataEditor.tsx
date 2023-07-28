import { PrimitiveAtom, useAtom } from "jotai";
import { focusAtom } from "jotai-optics";
import { useMemo } from "react";

import { Station, TripServiceInfo } from "@/api/protocol/motis";

import { MeasureUnion, measureTypeNeedsRecipients } from "@/data/measures";

import TripServiceInfoView from "@/components/TripServiceInfoView";
import StationPicker from "@/components/inputs/StationPicker";
import TimeInput from "@/components/inputs/TimeInput";
import TripPicker from "@/components/inputs/TripPicker";

export interface SharedDataEditorProps {
  measureAtom: PrimitiveAtom<MeasureUnion>;
}

const labelClass = "font-semibold";

function SharedDataEditor({ measureAtom }: SharedDataEditorProps): JSX.Element {
  const sharedAtom = useMemo(() => {
    console.log("SharedDataEditor: creating sharedAtom");
    return focusAtom(measureAtom, (optic) => optic.prop("shared"));
  }, [measureAtom]);
  const [shared, setShared] = useAtom(sharedAtom);
  const typeAtom = useMemo(
    () => focusAtom(measureAtom, (optic) => optic.prop("type")),
    [measureAtom],
  );
  const [measureType] = useAtom(typeAtom);

  const showRecipients = measureTypeNeedsRecipients(measureType);

  const addTrip = (trip: TripServiceInfo | undefined) => {
    if (trip) {
      const id = JSON.stringify(trip.trip);
      if (!shared.recipients.trips.some((t) => JSON.stringify(t.trip) === id)) {
        setShared((s) => {
          return {
            ...s,
            recipients: {
              ...s.recipients,
              trips: [...s.recipients.trips, trip],
            },
          };
        });
      }
    }
  };

  const removeTrip = (idx: number) => {
    setShared((s) => {
      const newTrips = [...s.recipients.trips];
      newTrips.splice(idx, 1);
      return { ...s, recipients: { ...s.recipients, trips: newTrips } };
    });
  };

  const addStation = (station: Station | undefined) => {
    if (
      station &&
      !shared.recipients.stations.some((s) => s.id === station.id)
    ) {
      setShared((s) => {
        return {
          ...s,
          recipients: {
            ...s.recipients,
            stations: [...s.recipients.stations, station],
          },
        };
      });
    }
  };

  const removeStation = (idx: number) => {
    setShared((s) => {
      const newStations = [...s.recipients.stations];
      newStations.splice(idx, 1);
      return { ...s, recipients: { ...s.recipients, stations: newStations } };
    });
  };

  const setTime = (date: Date) => {
    setShared((s) => {
      return { ...s, time: date };
    });
  };

  return (
    <div className="flex flex-col gap-4 mt-2 mb-5 pb-2">
      <div>
        <div className={labelClass}>Durchführung der Maßnahme</div>
        <div>
          <TimeInput
            value={shared.time}
            onChange={setTime}
            className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
          />
        </div>
      </div>
      {showRecipients && (
        <>
          <div>
            <div className={labelClass}>Ansage in Zug</div>
            <ul className="leading-loose">
              {shared.recipients.trips.map((tsi, idx) => (
                <li key={JSON.stringify(tsi.trip)}>
                  <TripServiceInfoView tsi={tsi} format="Short" />
                  <button
                    type="button"
                    className="ml-3 px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-xs rounded"
                    onClick={() => removeTrip(idx)}
                  >
                    Entfernen
                  </button>
                </li>
              ))}
            </ul>
            <TripPicker
              onTripPicked={addTrip}
              clearOnPick={true}
              longDistanceOnly={false}
            />
          </div>
          <div>
            <div className={labelClass}>Ansage an Station</div>
            <ul className="leading-loose">
              {shared.recipients.stations.map((station, idx) => (
                <li key={station.id}>
                  <span>{station.name}</span>
                  <button
                    type="button"
                    className="ml-3 px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-xs rounded"
                    onClick={() => removeStation(idx)}
                  >
                    Entfernen
                  </button>
                </li>
              ))}
            </ul>
            <StationPicker
              onStationPicked={addStation}
              clearOnPick={true}
              clearButton={false}
            />
          </div>
        </>
      )}
    </div>
  );
}

export default SharedDataEditor;
