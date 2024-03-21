import { useAtom } from "jotai";
import React, { ReactNode } from "react";
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
import { formatNumber } from "@/data/numberFormat";

import { EMTPY_CAPACITY_DATA, addCapacityData } from "@/util/capacity";
import {
  getCapacitySourceShortText,
  getCapacitySourceTooltip,
  getFormationCapacitySourceShortText,
} from "@/util/capacitySource";
import { formatDate, formatTime } from "@/util/dateFormat";

import TripServiceInfoView from "@/components/TripServiceInfoView";
import Baureihe from "@/components/util/Baureihe";

import { cn } from "@/lib/utils";

interface CapacityInfoProps {
  tripId: TripId;
}

function CapacityInfo({ tripId }: CapacityInfoProps): ReactNode {
  const [universe] = useAtom(universeAtom);
  const { data, isPending, error } = usePaxMonGetTripCapacity({
    universe,
    trips: [tripId],
  });

  if (!data) {
    if (isPending) {
      return <div>Kapazitätsinformationen werden geladen...</div>;
    } else {
      return (
        <div>
          Fehler beim Laden der Kapazitätsinformationen:{" "}
          {error instanceof Error ? error.message : `Unbekannter Fehler`}
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
      <h1 className="mb-4 text-2xl">Kapazitätsinformationen</h1>
      <div>
        {trips.map((t) => (
          <TripCapacityInfo data={t} key={JSON.stringify(t.tsi.trip)} />
        ))}
      </div>
      <div>
        <div>Verfügbare Kapazitätsdaten:</div>
        <div>
          Basierend auf Wagenreihungen:{" "}
          {formatNumber(data.trip_formation_map_size)} Wagenreihungen,{" "}
          {formatNumber(data.vehicle_capacity_map_size)} Fahrzeuge,{" "}
          {formatNumber(data.vehicle_group_capacity_map_size)} Fahrzeuggruppen,{" "}
          {formatNumber(data.gattung_capacity_map_size)} Fahrzeuggattungen,{" "}
          {formatNumber(data.baureihe_capacity_map_size)} Baureihen
        </div>
        <div>
          Basierend auf Fahrten: {formatNumber(data.trip_capacity_map_size)}{" "}
          Zugkapazitäten, {formatNumber(data.category_capacity_map_size)}{" "}
          Kapazitäten für Zugkategorien
        </div>
      </div>
    </div>
  );
}

function TripCapacityInfo({ data }: { data: PaxMonTripCapacityInfo }) {
  return (
    <div className="mb-16">
      <div className="sticky -top-2 flex items-center gap-4">
        <TripServiceInfoView
          tsi={data.tsi}
          format={"Short"}
          link={true}
          className="text-xl"
        />
        <span>{formatDate(data.tsi.trip.time)}</span>
        <span>
          {data.tsi.primary_station.name} → {data.tsi.secondary_station.name}
        </span>
      </div>
      <div className="ml-4 mt-2 flex flex-col gap-8">
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
          <div className="flex w-24 shrink-0 justify-between">
            <span className="w-1/2 text-gray-600">
              {formatTime(section.departure_schedule_time)}
            </span>
            <span
              className={cn(
                "w-1/2",
                departureDelayed ? "text-red-600" : "text-green-600",
              )}
            >
              {formatTime(section.departure_current_time)}
            </span>
          </div>
          <div className="truncate hover:relative hover:overflow-visible">
            {section.from.name}
          </div>
        </div>
        <div className="flex gap-1">
          <div className="flex w-24 shrink-0 justify-between">
            <span className="w-1/2 text-gray-600">
              {formatTime(section.arrival_schedule_time)}
            </span>
            <span
              className={cn(
                "w-1/2",
                arrivalDelayed ? "text-red-600" : "text-green-600",
              )}
            >
              {formatTime(section.arrival_current_time)}
            </span>
          </div>
          <div className="truncate hover:relative hover:overflow-visible">
            {section.to.name}
          </div>
        </div>
      </div>
      <div className="flex w-20 flex-col items-center">
        <div className="font-semibold">
          {section.capacity_type === "Known" ? section.capacity.seats : "?"}
        </div>
        <div
          className="text-xs"
          title={getCapacitySourceTooltip(section.capacity_source)}
        >
          {getCapacitySourceShortText(section.capacity_source)}
        </div>
      </div>
      <div className="flex grow flex-col gap-4">
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
          <div className="flex flex-wrap gap-1 text-db-red-500">
            <span className="font-semibold">
              Manuell überschriebene Kapazität:
            </span>
            <span>{mt.override[0].seats}</span>
          </div>
        )}
        {has_trip_lookup && (
          <div className="flex flex-wrap gap-1">
            <span className="font-semibold">Zugkapazität:</span>
            <span>{mt.trip_lookup_capacity.seats}</span>
            <span
              title={getCapacitySourceTooltip(mt.trip_lookup_capacity_source)}
            >
              ({getCapacitySourceShortText(mt.trip_lookup_capacity_source)})
            </span>
          </div>
        )}
        {has_trip_formation && (
          <>
            <div className="flex flex-wrap gap-1">
              <span className="font-semibold">Kapazität aus Wagenreihung:</span>
              <span>{mt.trip_formation_capacity.seats}</span>
              <span>
                {`(Kapazitätsdaten: ${getFormationCapacitySourceShortText(
                  mt.trip_formation_capacity_source,
                  false,
                )})`}
              </span>
            </div>
            <SectionVehicleGroups mt={mt} />
          </>
        )}
        {!has_capacity && <div>Keine Kapazitätsinformationen gefunden</div>}
      </div>
    </div>
  );
}

function SectionVehicleGroups({ mt }: { mt: PaxMonMergedTripCapacityInfo }) {
  const capacitySum = mt.vehicle_groups.reduce(
    (sum, vg) =>
      vg.capacity.length === 1 ? addCapacityData(sum, vg.capacity[0]) : sum,
    EMTPY_CAPACITY_DATA,
  );

  return (
    <div>
      <table className="mt-2">
        <thead>
          <tr className="border-b-2 border-db-cool-gray-300 text-sm font-semibold">
            <td className="px-2">Fahrzeuggruppe</td>
            <td className="px-2">Zugnummer</td>
            <td className="px-2">von</td>
            <td className="px-2">bis</td>
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
          </tr>
        </thead>
        <tbody>
          {mt.vehicle_groups.map((vg) => (
            <tr
              key={vg.name}
              className={cn(
                vg.capacity.length === 1
                  ? "text-green-600"
                  : "text-db-cool-gray-500",
              )}
            >
              <td className="px-2">{vg.name}</td>
              <td className="px-2">{vg.primary_trip_id.train_nr}</td>
              <td className="px-2">{vg.start.name}</td>
              <td className="px-2">{vg.destination.name}</td>
              <td className="px-2 text-center">
                {vg.capacity.length === 1 ? vg.capacity[0].seats : 0}
              </td>
              <td className="px-2 text-center">
                {vg.capacity.length === 1 ? vg.capacity[0].seats_1st : 0}
              </td>
              <td className="px-2 text-center">
                {vg.capacity.length === 1 ? vg.capacity[0].seats_2nd : 0}
              </td>
              <td className="px-2 text-center">
                {vg.capacity.length === 1 ? vg.capacity[0].standing : 0}
              </td>
              <td className="px-2 text-center">
                {vg.capacity.length === 1 ? vg.capacity[0].total_limit : 0}
              </td>
              <td className="px-2 text-center">
                {vg.capacity.length === 1 ? vg.capacity[0].limit : 0}
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr className="border-t-2 border-db-cool-gray-300 font-semibold">
            <td className="px-2" colSpan={4}></td>
            <td className="px-2 text-center">{capacitySum.seats}</td>
            <td className="px-2 text-center">{capacitySum.seats_1st}</td>
            <td className="px-2 text-center">{capacitySum.seats_2nd}</td>
            <td className="px-2 text-center">{capacitySum.standing}</td>
            <td className="px-2 text-center">{capacitySum.total_limit}</td>
            <td className="px-2 text-center">{capacitySum.limit}</td>
            <td className="px-2"></td>
          </tr>
        </tfoot>
      </table>
      <div>
        {mt.vehicle_groups.map((vg) => (
          <VehicleGroup key={vg.name} vg={vg} />
        ))}
      </div>
    </div>
  );
}

function VehicleGroup({ vg }: { vg: PaxMonVehicleGroupInfo }) {
  const capacitySum = vg.vehicles.reduce(
    (sum, v) => addCapacityData(sum, v.data),
    EMTPY_CAPACITY_DATA,
  );

  return (
    <div className="mt-2">
      <div className="font-semibold">Fahrzeuggruppe {vg.name}:</div>
      <table className="mt-2">
        <thead>
          <tr className="border-b-2 border-db-cool-gray-300 text-sm font-semibold">
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
            <td className="px-2 text-center">Quelle</td>
          </tr>
        </thead>
        <tbody>
          {vg.vehicles.map((v, idx) => (
            <tr
              key={idx}
              className={cn(
                v.guessed ? "text-fuchsia-500" : !v.uic_found && "text-red-500",
              )}
            >
              <td className="px-2">{v.order}</td>
              <td className="px-2">{v.type_code}</td>
              <td className="px-2">
                <Baureihe baureihe={v.baureihe} />
              </td>
              <td className="px-2">{v.uic != 0 ? v.uic : ""}</td>
              <td className="px-2 text-center">{v.data.seats}</td>
              <td className="px-2 text-center">{v.data.seats_1st}</td>
              <td className="px-2 text-center">{v.data.seats_2nd}</td>
              <td className="px-2 text-center">{v.data.standing}</td>
              <td className="px-2 text-center">{v.data.total_limit}</td>
              <td className="px-2 text-center">{v.data.limit}</td>
              <td className="px-2 text-center">
                {getFormationCapacitySourceShortText(v.capacity_source, true)}
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr className="border-t-2 border-db-cool-gray-300 font-semibold">
            <td className="px-2" colSpan={4}></td>
            <td className="px-2 text-center">{capacitySum.seats}</td>
            <td className="px-2 text-center">{capacitySum.seats_1st}</td>
            <td className="px-2 text-center">{capacitySum.seats_2nd}</td>
            <td className="px-2 text-center">{capacitySum.standing}</td>
            <td className="px-2 text-center">{capacitySum.total_limit}</td>
            <td className="px-2 text-center">{capacitySum.limit}</td>
            <td className="px-2"></td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

export default CapacityInfo;
