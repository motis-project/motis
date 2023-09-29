import { ReactNode } from "react";
import { Link } from "react-router-dom";

import { TripServiceInfo } from "@/api/protocol/motis";

import { formatDateTime } from "@/util/dateFormat";

import { cn } from "@/lib/utils";

interface TripServiceInfoViewProps {
  tsi: TripServiceInfo;
  format: "Short" | "ShortAll" | "Long";
  link?: boolean;
  className?: string | undefined;
}

function TripServiceInfoView({
  tsi,
  format,
  link = false,
  className = undefined,
}: TripServiceInfoViewProps): ReactNode {
  const names = [
    ...new Set(tsi.service_infos.map((si) => `${si.category} ${si.train_nr}`)),
  ];

  let content;
  if (format === "Short") {
    content = (
      <span className={cn(className)}>{names[0] ?? tsi.trip.train_nr}</span>
    );
  } else if (format === "ShortAll") {
    content = (
      <span className={cn(className)}>
        {names.length > 0 ? names.join(", ") : `${tsi.trip.train_nr}`}
      </span>
    );
  } else {
    content = (
      <div className={cn("w-full", className)}>
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

  return link ? (
    <Link to={`/trips/${encodeURIComponent(JSON.stringify(tsi.trip))}`}>
      {content}
    </Link>
  ) : (
    content
  );
}

export default TripServiceInfoView;
