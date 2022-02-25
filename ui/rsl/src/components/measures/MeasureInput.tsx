import React, { useState } from "react";

import { queryKeys, usePaxMonStatusQuery } from "../../api/paxmon";
import { Station, TripServiceInfo } from "../../api/protocol/motis";
import {
  LoadLevel,
  MeasureType,
  MeasureWrapper,
} from "../../api/protocol/motis/paxforecast";

import TripPicker from "../TripPicker";
import StationPicker from "../StationPicker";
import TripServiceInfoView from "../TripServiceInfoView";
import TimeInput from "./TimeInput";
import { useMutation, useQueryClient } from "react-query";
import { useAtom } from "jotai";
import { universeAtom } from "../../data/simulation";
import { sendPaxForecastApplyMeasuresRequest } from "../../api/paxforecast";

const loadLevels: Array<{ level: LoadLevel; label: string }> = [
  { level: "Unknown", label: "unbekannt" },
  { level: "Low", label: "gering" },
  { level: "NoSeats", label: "keine Sitzplätze mehr" },
  { level: "Full", label: "voll" },
];

const measureTypes: Array<{ type: MeasureType; label: string }> = [
  { type: "TripLoadInfoMeasure", label: "Auslastungsinformation" },
  { type: "TripRecommendationMeasure", label: "Alternativenempfehlung" },
];

const labelClass = "font-semibold";

function MeasureInput(): JSX.Element {
  const queryClient = useQueryClient();
  const [universe] = useAtom(universeAtom);
  const [recipientTrips, setRecipientTrips] = useState<TripServiceInfo[]>([]);
  const [recipientStations, setRecipientStations] = useState<Station[]>([]);
  const { data: status } = usePaxMonStatusQuery(universe);
  const [time, setTime] = useState(() =>
    status ? new Date(status.system_time * 1000) : new Date()
  );

  const [measureType, setMeasureType] = useState<MeasureType>(
    "TripLoadInfoMeasure"
  );

  const [loadInfoTrip, setLoadInfoTrip] = useState<TripServiceInfo>();
  const [loadInfoLevel, setLoadInfoLevel] = useState<LoadLevel>("Unknown");

  const [tripRecDestination, setTripRecDestination] = useState<Station>();
  const [tripRecInterchange, setTripRecInterchange] = useState<Station>();
  const [tripRecTrip, setTripRecTrip] = useState<TripServiceInfo>();

  const applyEnabled = universe != 0;

  const applyMeasuresMutation = useMutation(
    (measures: MeasureWrapper[]) =>
      sendPaxForecastApplyMeasuresRequest({
        universe,
        measures,
        replace_existing: true,
        preparation_time: 0,
      }),
    {
      onSuccess: async () => {
        console.log("measures applied");
        await queryClient.invalidateQueries(queryKeys.trip());
      },
    }
  );

  const addTrip = (trip: TripServiceInfo | undefined) => {
    if (trip) {
      const id = JSON.stringify(trip.trip);
      if (!recipientTrips.some((t) => JSON.stringify(t.trip) === id)) {
        setRecipientTrips([...recipientTrips, trip]);
      }
    }
  };

  const removeTrip = (idx: number) => {
    const newTrips = [...recipientTrips];
    newTrips.splice(idx, 1);
    setRecipientTrips(newTrips);
  };

  const addStation = (station: Station | undefined) => {
    if (station && !recipientStations.some((s) => s.id == station.id)) {
      setRecipientStations([...recipientStations, station]);
    }
  };

  const removeStation = (idx: number) => {
    const newStations = [...recipientStations];
    newStations.splice(idx, 1);
    setRecipientStations(newStations);
  };

  const buildMeasure: () => MeasureWrapper = () => {
    const recipients = {
      trips: recipientTrips.map((tsi) => tsi.trip),
      stations: recipientStations.map((s) => s.id),
    };
    const unixTime = Math.round(time.getTime() / 1000);

    if (recipients.trips.length == 0 && recipients.stations.length == 0) {
      throw new Error("Kein Ort für die Ansage ausgewählt");
    }

    switch (measureType) {
      case "TripLoadInfoMeasure":
        if (!loadInfoTrip) {
          throw new Error("Kein Trip ausgewählt");
        }
        return {
          measure_type: "TripLoadInfoMeasure",
          measure: {
            recipients,
            time: unixTime,
            trip: loadInfoTrip.trip,
            level: loadInfoLevel,
          },
        };
      case "TripRecommendationMeasure":
        if (!tripRecDestination || !tripRecInterchange || !tripRecTrip) {
          throw new Error("Nicht alle benötigten Felder ausgefüllt");
        }
        return {
          measure_type: "TripRecommendationMeasure",
          measure: {
            recipients,
            time: unixTime,
            planned_trips: [],
            planned_destinations: [tripRecDestination.id],
            planned_long_distance_destinations: [],
            recommended_trip: tripRecTrip.trip,
            interchange_station: tripRecInterchange.id,
          },
        };
    }
  };

  const measureDetails: () => JSX.Element = () => {
    switch (measureType) {
      case "TripLoadInfoMeasure":
        return (
          <>
            {/*<div className="font-semibold mt-2">Auslastungsinformation</div>*/}
            <div>
              <div className={labelClass}>Trip</div>
              <div>
                <TripPicker
                  onTripPicked={setLoadInfoTrip}
                  clearOnPick={false}
                  longDistanceOnly={false}
                />
              </div>
            </div>
            <div>
              <div className={labelClass}>Auslastungsstufe</div>
              <div className="flex flex-col">
                {loadLevels.map(({ level, label }) => (
                  <label key={level} className="inline-flex items-center gap-1">
                    <input
                      type="radio"
                      name="load-level"
                      value={level}
                      checked={loadInfoLevel == level}
                      onChange={() => setLoadInfoLevel(level)}
                    />
                    {label}
                  </label>
                ))}
              </div>
            </div>
          </>
        );
      case "TripRecommendationMeasure":
        return (
          <>
            {/*<div className="font-semibold mt-2">Alternativenempfehlung</div>*/}
            <div>
              <div className={labelClass}>Reisende Richtung</div>
              <StationPicker
                onStationPicked={setTripRecDestination}
                clearOnPick={false}
              />
            </div>
            <div>
              <div className={labelClass}>Umsteigen an Station</div>
              <StationPicker
                onStationPicked={setTripRecInterchange}
                clearOnPick={false}
              />
            </div>
            <div>
              <div className={labelClass}>in Trip</div>
              <TripPicker
                onTripPicked={setTripRecTrip}
                clearOnPick={false}
                longDistanceOnly={false}
              />
            </div>
          </>
        );
    }
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        try {
          const measure = buildMeasure();
          console.log(JSON.stringify(measure, null, 2));
          applyMeasuresMutation.mutate([measure]);
        } catch (ex) {
          alert(ex);
        }
      }}
    >
      <div className="flex flex-col gap-4">
        <div>
          <div className={labelClass}>Ansage in Zug</div>
          <ul className="leading-loose">
            {recipientTrips.map((tsi, idx) => (
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
            {recipientStations.map((station, idx) => (
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
          <StationPicker onStationPicked={addStation} clearOnPick={true} />
        </div>
        <div>
          <div className={labelClass}>Zeitpunkt der Ansage</div>
          <div>
            <TimeInput
              value={time}
              onChange={setTime}
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
            />
          </div>
        </div>

        <div>
          <div className={labelClass}>Maßnahmentyp</div>
          <div className="flex flex-col">
            {measureTypes.map(({ type, label }) => (
              <label key={type} className="inline-flex items-center gap-1">
                <input
                  type="radio"
                  name="measure-type"
                  value={type}
                  checked={measureType == type}
                  onChange={() => setMeasureType(type)}
                />
                {label}
              </label>
            ))}
          </div>
        </div>

        {measureDetails()}

        <button
          className={`w-full p-3 rounded ${
            applyMeasuresMutation.isLoading
              ? "bg-db-red-300 text-db-red-100 cursor-wait"
              : applyEnabled
              ? "bg-db-red-500 hover:bg-db-red-600 text-white"
              : "bg-db-red-300 text-db-red-100 cursor-not-allowed"
          }`}
          disabled={!applyEnabled}
        >
          Maßnahme simulieren
        </button>

        {applyMeasuresMutation.isError && (
          <div>
            <div className={labelClass}>
              Fehler bei der Maßnahmensimulation:
            </div>
            <div>
              {applyMeasuresMutation.error instanceof Error
                ? applyMeasuresMutation.error.message
                : "Unbekannter Fehler"}
            </div>
          </div>
        )}

        {applyEnabled ? null : (
          <div>
            Maßnahmen können nicht im Hauptuniversum (#0) simuliert werden.
            Bitte zuerst ein neues Paralleluniversum anlegen (Kopieren-Button in
            der Kopfzeile) bzw. auswählen.
          </div>
        )}
      </div>
    </form>
  );
}

export default MeasureInput;
