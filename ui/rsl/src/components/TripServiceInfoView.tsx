import { TripServiceInfo } from "@/api/protocol/motis";

import { formatDateTime } from "@/util/dateFormat";

interface TripServiceInfoViewProps {
  tsi: TripServiceInfo;
  format: "Short" | "ShortAll" | "Long";
}

function TripServiceInfoView({
  tsi,
  format,
}: TripServiceInfoViewProps): JSX.Element {
  const names = [
    ...new Set(tsi.service_infos.map((si) => `${si.category} ${si.train_nr}`)),
  ];
  if (format === "Short") {
    return <span>{names[0] ?? tsi.trip.train_nr}</span>;
  } else if (format === "ShortAll") {
    return (
      <span>
        {names.length > 0 ? names.join(", ") : `${tsi.trip.train_nr}`}
      </span>
    );
  } else {
    return (
      <div className="w-full">
        <div className="text-lg">
          {names.length > 0 ? names.join(", ") : `${tsi.trip.train_nr}`}
        </div>
        <div className="flex justify-between text-sm">
          <div className="flex flex-col items-start">
            <div>{tsi.primary_station.name}</div>
            <div>{formatDateTime(tsi.trip.time)}</div>
          </div>
          <div className="flex flex-col items-end">
            <div>{tsi.secondary_station.name}</div>
            <div>{formatDateTime(tsi.trip.target_time)}</div>
          </div>
        </div>
      </div>
    );
  }
}

export default TripServiceInfoView;
