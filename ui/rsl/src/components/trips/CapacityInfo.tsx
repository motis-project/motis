import { useAtom } from "jotai";
import React from "react";
import { Link } from "react-router-dom";

import { TripId } from "@/api/protocol/motis";
import {
  PaxMonMergedTripCapacityInfo,
  PaxMonSectionCapacityInfo,
  PaxMonTripCapacityInfo,
  PaxMonVehicleGroupInfo,
} from "@/api/protocol/motis/paxmon";

import { usePaxMonGetTripCapacity } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";

import {
  getCapacitySourceShortText,
  getCapacitySourceTooltip,
} from "@/util/capacitySource";
import classNames from "@/util/classNames";
import { formatDate, formatTime } from "@/util/dateFormat";

import TripServiceInfoView from "@/components/TripServiceInfoView";

type CapacityInfoProps = {
  tripId: TripId;
};

function CapacityInfo({ tripId }: CapacityInfoProps): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const { data, isLoading, error } = usePaxMonGetTripCapacity({
    universe,
    trips: [tripId],
  });

  if (!data) {
    if (isLoading) {
      return <div>Kapazitätsinformationen werden geladen...</div>;
    } else {
      return (
        <div>
          Fehler beim Laden der Kapazitätsinformationen:{" "}
          {error instanceof Error ? error.message : `${error}`}
        </div>
      );
    }
  }

  const mainTrip = JSON.stringify(tripId);
  const trips = [...data.trips];
  trips.sort((a, b) => {
    const aTrip = JSON.stringify(a.tsi.trip);
    const bTrip = JSON.stringify(b.tsi.trip);
    if (aTrip == bTrip) {
      return 0;
    } else if (bTrip == mainTrip) {
      return 1;
    } else if (aTrip == mainTrip) {
      return -1;
    } else {
      return 0;
    }
  });

  return (
    <div className="py-4">
      <h1 className="text-2xl mb-4">Kapazitätsinformationen</h1>
      <div>
        {trips.map((t) => (
          <TripCapacityInfo data={t} key={JSON.stringify(t.tsi.trip)} />
        ))}
      </div>
      <div>
        <div>Verfügbare Kapazitätsdaten:</div>
        <div>
          {data.vehicle_capacity_map_size} Fahrzeuge,{" "}
          {data.trip_formation_map_size} Wagenreihungen,{" "}
          {data.trip_capacity_map_size} Zugkapazitäten,{" "}
          {data.category_capacity_map_size} Kapazitäten für Zugkategorien
        </div>
      </div>
    </div>
  );
}

function TripCapacityInfo({ data }: { data: PaxMonTripCapacityInfo }) {
  return (
    <div className="mb-16">
      <div className="flex gap-4 items-center sticky -top-2 bg-db-cool-gray-100">
        <Link
          to={`/trips/${encodeURIComponent(JSON.stringify(data.tsi.trip))}`}
          className="text-xl"
        >
          <TripServiceInfoView tsi={data.tsi} format={"Short"} />
        </Link>
        <span>{formatDate(data.tsi.trip.time)}</span>
        <span>
          {data.tsi.primary_station.name} → {data.tsi.secondary_station.name}
        </span>
      </div>
      <div className="flex flex-col gap-8 mt-2 ml-4">
        {data.sections.map((section, idx) => (
          <SectionCapacityInfo section={section} key={idx} />
        ))}
      </div>
    </div>
  );
}

function SectionCapacityInfo({
  section,
}: {
  section: PaxMonSectionCapacityInfo;
}) {
  const departureDelayed =
    section.departure_current_time > section.departure_schedule_time;
  const arrivalDelayed =
    section.arrival_current_time > section.arrival_schedule_time;

  return (
    <div className="flex gap-1">
      <div className="w-80 shrink-0">
        <div className="flex gap-1">
          <div className="flex justify-between w-24 shrink-0">
            <span className="text-gray-600 w-1/2">
              {formatTime(section.departure_schedule_time)}
            </span>
            <span
              className={classNames(
                "w-1/2",
                departureDelayed ? "text-red-600" : "text-green-600"
              )}
            >
              {formatTime(section.departure_current_time)}
            </span>
          </div>
          <div className="truncate hover:overflow-visible hover:relative">
            {section.from.name}
          </div>
        </div>
        <div className="flex gap-1">
          <div className="flex justify-between w-24 shrink-0">
            <span className="text-gray-600 w-1/2">
              {formatTime(section.arrival_schedule_time)}
            </span>
            <span
              className={classNames(
                "w-1/2",
                arrivalDelayed ? "text-red-600" : "text-green-600"
              )}
            >
              {formatTime(section.arrival_current_time)}
            </span>
          </div>
          <div className="truncate hover:overflow-visible hover:relative">
            {section.to.name}
          </div>
        </div>
      </div>
      <div className="w-20 flex flex-col items-center">
        <div className="font-semibold">
          {section.capacity_type === "Known" ? section.capacity : "?"}
        </div>
        <div
          className="text-xs"
          title={getCapacitySourceTooltip(section.capacity_source)}
        >
          {getCapacitySourceShortText(section.capacity_source)}
        </div>
      </div>
      <div className="grow flex flex-col gap-4">
        {section.merged_trips.map((mt, idx) => (
          <MergedTripCapacityInfo mt={mt} key={idx} />
        ))}
      </div>
    </div>
  );
}

function MergedTripCapacityInfo({ mt }: { mt: PaxMonMergedTripCapacityInfo }) {
  const has_trip_lookup = mt.trip_lookup_capacity_source !== "Unknown";
  const has_trip_formation = mt.trip_formation_found;
  const has_override = mt.override.length === 1;
  const has_capacity = has_trip_lookup || has_trip_formation || has_override;

  return (
    <div className="flex gap-2">
      <div className="w-24">
        <Link to={`/trips/${encodeURIComponent(JSON.stringify(mt.trip))}`}>
          {mt.service_info.category} {mt.service_info.train_nr}:
        </Link>
      </div>
      <div className="flex flex-col gap-1">
        {has_override && (
          <div className="flex gap-1 flex-wrap text-db-red-500">
            <span className="font-semibold">
              Manuell überschriebene Kapazität:
            </span>
            <span>{mt.override[0].seats}</span>
          </div>
        )}
        {has_trip_lookup && (
          <div className="flex gap-1 flex-wrap">
            <span className="font-semibold">Zugkapazität:</span>
            <span>{mt.trip_lookup_capacity}</span>
            <span
              title={getCapacitySourceTooltip(mt.trip_lookup_capacity_source)}
            >
              ({getCapacitySourceShortText(mt.trip_lookup_capacity_source)})
            </span>
          </div>
        )}
        {has_trip_formation && (
          <>
            <div className="flex gap-1 flex-wrap">
              <span className="font-semibold">Kapazität aus Wagenreihung:</span>
              <span>{mt.trip_formation_capacity.seats}</span>
            </div>
            <SectionVehicles mt={mt} />
          </>
        )}
        {!has_capacity && <div>Keine Kapazitätsinformationen gefunden</div>}
      </div>
    </div>
  );
}

function SectionVehicles({ mt }: { mt: PaxMonMergedTripCapacityInfo }) {
  return (
    <div>
      <table>
        <thead>
          <tr className="text-sm font-semibold border-b-2 border-db-cool-gray-300">
            <td className="px-2">Wagen</td>
            <td className="px-2">Bauart</td>
            <td className="px-2">Baureihe</td>
            <td className="px-2">UIC-Wagennummer</td>
            <td className="px-2 text-center" title="Sitzplätze insgesamt">
              Sitze
            </td>
            <td className="px-2 text-center" title="Sitzplätze 1. Klasse">
              1. Kl
            </td>
            <td className=" text-center" title="Sitzplätze 2. Klasse">
              2. Kl
            </td>
            <td className="px-2 text-center" title="Stehplätze">
              Steh.
            </td>
            <td
              className="px-2 text-center"
              title="Zulässige Gesamtanzahl Reisender"
            >
              Zul.
            </td>
            <td
              className="px-2 text-center"
              title="Maximalkapazität (Zulässige Gesamtanzahl Reisender oder Anzahl Sitzplätze)"
            >
              Max.
            </td>
            <td className="px-2">Wagengruppen</td>
          </tr>
        </thead>
        <tbody>
          {mt.vehicles.map((v) => (
            <tr key={v.uic} className={classNames(!v.found && "text-red-500")}>
              <td className="px-2">{v.order}</td>
              <td className="px-2">{v.type_code}</td>
              <td className="px-2">
                <Baureihe baureihe={v.baureihe} />
              </td>
              <td className="px-2">{v.uic}</td>
              <td className="px-2 text-center">{v.data.seats}</td>
              <td className="px-2 text-center">{v.data.seats_1st}</td>
              <td className="px-2 text-center">{v.data.seats_2nd}</td>
              <td className="px-2 text-center">{v.data.standing}</td>
              <td className="px-2 text-center">{v.data.total_limit}</td>
              <td className="px-2 text-center">{v.data.limit}</td>
              <td className="px-2">
                {v.vehicle_groups
                  .map((idx) => mt.vehicle_groups[idx].name)
                  .join(", ")}
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr className="font-semibold border-t-2 border-db-cool-gray-300">
            <td className="px-2" colSpan={4}></td>
            <td className="px-2 text-center">
              {mt.trip_formation_capacity.seats}
            </td>
            <td className="px-2 text-center">
              {mt.trip_formation_capacity.seats_1st}
            </td>
            <td className="px-2 text-center">
              {mt.trip_formation_capacity.seats_2nd}
            </td>
            <td className="px-2 text-center">
              {mt.trip_formation_capacity.standing}
            </td>
            <td className="px-2 text-center">
              {mt.trip_formation_capacity.total_limit}
            </td>
            <td className="px-2 text-center">
              {mt.trip_formation_capacity.limit}
            </td>
            <td className="px-2"></td>
          </tr>
        </tfoot>
      </table>
      <div className="flex flex-col gap-2">
        {mt.vehicle_groups.map((vg) => (
          <VehicleGroup vg={vg} key={vg.name} />
        ))}
      </div>
    </div>
  );
}

function VehicleGroup({ vg }: { vg: PaxMonVehicleGroupInfo }) {
  return (
    <div>
      Wagengruppe {vg.name}: Zug {vg.primary_trip_id.train_nr}, von{" "}
      {vg.start.name} bis {vg.destination.name}
    </div>
  );
}

function Baureihe({ baureihe }: { baureihe: string }) {
  const m = /^[ITR](\d{3})([0-9A-Z])$/.exec(baureihe);
  if (m) {
    return (
      <span title={baureihe}>
        {m[1]}.{m[2]}
      </span>
    );
  } else {
    return <span>{baureihe}</span>;
  }
}

export default CapacityInfo;
