import React, { useState } from "react";

import { usePaxMonStatusQuery } from "../../api/paxmon";
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

type MeasureInputProps = {
  onAddMeasure: (measure: MeasureWrapper) => void;
};

function MeasureInput(): JSX.Element {
  const queryClient = useQueryClient();
  const [universe] = useAtom(universeAtom);
  const [recipientTrips, setRecipientTrips] = useState<TripServiceInfo[]>([]);
  const [recipientStations, setRecipientStations] = useState<Station[]>([]);
  const { data: status } = usePaxMonStatusQuery();
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
        await queryClient.invalidateQueries(["paxmon", "trip"]); // TODO
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
          <div>
            {/*<div className="font-semibold mt-2">Auslastungsinformation</div>*/}
            <div>Trip</div>
            <div>
              <TripPicker
                onTripPicked={setLoadInfoTrip}
                clearOnPick={false}
                longDistanceOnly={false}
              />
            </div>
            <div>Auslastungsstufe</div>
            <div className="flex gap-4">
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
        );
      case "TripRecommendationMeasure":
        return (
          <div>
            {/*<div className="font-semibold mt-2">Alternativenempfehlung</div>*/}
            <div>Reisende Richtung</div>
            <div>
              <StationPicker
                onStationPicked={setTripRecDestination}
                clearOnPick={false}
              />
            </div>
            <div>Umsteigen an Station</div>
            <div>
              <StationPicker
                onStationPicked={setTripRecInterchange}
                clearOnPick={false}
              />
            </div>
            <div>in Trip</div>
            <div>
              <TripPicker
                onTripPicked={setTripRecTrip}
                clearOnPick={false}
                longDistanceOnly={false}
              />
            </div>
          </div>
        );
    }
  };

  return (
    <div className="bg-blue-100">
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
        <div>
          <div>Ansage in Zug</div>
          <ul>
            {recipientTrips.map((tsi, idx) => (
              <li key={JSON.stringify(tsi.trip)}>
                <TripServiceInfoView tsi={tsi} format="Long" />
                <button
                  type="button"
                  className="ml-3 px-2 py-1 bg-red-200 hover:bg-red-100 text-xs rounded-xl"
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
          <div>Ansage an Station</div>
          <ul>
            {recipientStations.map((station, idx) => (
              <li key={station.id}>
                <span>{station.name}</span>
                <button
                  type="button"
                  className="ml-3 px-2 py-1 bg-red-200 hover:bg-red-100 text-xs rounded-xl"
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
          <div>Zeitpunkt der Ansage</div>
          <div>
            <TimeInput value={time} onChange={setTime} />
          </div>
        </div>

        <div>
          <div>Maßnahmentyp</div>
          <div className="flex gap-4">
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

        <div>
          <button>Maßnahme simulieren</button>
        </div>
      </form>
    </div>
  );
}

export default MeasureInput;
