export interface TripId {
  // primary
  station_id: string;
  train_nr: number;
  time: number;

  // secondary
  target_station_id: string;
  target_time: number;
  line_id: string;
}

export interface Position {
  lat: number;
  lng: number;
}

export interface Station {
  id: string;
  name: string;
  pos: Position;
}

export interface ServiceInfo {
  name: string;
  category: string;
  train_nr: number;
  line: string;
  provider: string;
  clasz: number;
}

export interface TripServiceInfo {
  trip: TripId;
  primary_station: Station;
  secondary_station: Station;
  service_infos: ServiceInfo[];
}
