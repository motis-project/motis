import { TripServiceInfo } from "@/api/protocol/motis";

import { formatDateTime } from "@/util/dateFormat";

type TripServiceInfoViewProps = {
  tsi: TripServiceInfo;
  format: "Short" | "Long";
};

function TripServiceInfoView({
  tsi,
  format,
}: TripServiceInfoViewProps): JSX.Element {
  const names = [
    ...new Set(
      tsi.service_infos.map(
        (si) => `${si.category} ${si.train_nr}`
        //+ (si.line ? ` [Linie ${si.line}]` : "")
      )
    ),
  ];
  if (format === "Short") {
    return <span>{names.join(", ")}</span>;
  } else {
    return (
      <div className="w-full">
        <div className="text-lg">{names.join(", ")}</div>
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
