import { formatDateTime } from "../util/dateFormat";
import { TripServiceInfo } from "../api/protocol/motis";

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
      tsi.service_infos.map((si) =>
        si.line ? `${si.train_nr} [${si.name}]` : si.name
      )
    ),
  ];
  if (format === "Short") {
    return <span>{names.join(", ")}</span>;
  } else {
    return (
      <span>
        {names.join(", ")} ({tsi.primary_station.name} (
        {formatDateTime(tsi.trip.time)}) – {tsi.secondary_station.name} (
        {formatDateTime(tsi.trip.target_time)}))
      </span>
    );
  }
}

export default TripServiceInfoView;
