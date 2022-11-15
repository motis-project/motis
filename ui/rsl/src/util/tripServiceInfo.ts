import { TripServiceInfo } from "@/api/protocol/motis";

export function shortTripServiceName(tsi: TripServiceInfo): string {
  if (tsi.service_infos.length > 0) {
    const si = tsi.service_infos[0];
    return `${si.category} ${si.train_nr}`;
  } else {
    return `${tsi.trip.train_nr}`;
  }
}
