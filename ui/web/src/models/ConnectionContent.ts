/* eslint-disable camelcase */
import Interval from "./SmallTypes/Interval";
import Position from "./SmallTypes/Position";
import TripResponseContent from "./TripResponseContent";

export default interface ConnectionResponseContent {
  connections: TripResponseContent[]
}


export interface ConnectionRequestContent {
  start_type: "PretripStart" | "IntermodalPretripStart",
  start: Start,
  start_modes: Mode[],
  destination_type: "InputStation" | "InputPosition",
  destination: Position | InputStation
  destination_modes: Mode[]
}

export interface InputStation {
  name: string,
  id: string
}

export interface Start {
  position?: Position
  station?: InputStation,
  interval: Interval,
  min_connection_count: number,
  extend_interval_earlier: boolean,
  extend_interval_later: boolean
}

export interface Mode {
  mode_type: "FootPPR" | "Bike" | "Car" | "CarParking",
  mode: FootMode | BikeMode | CarMode | CarParkingMode
}

interface FootMode {
  search_options: {
    profile: "default" | "accessibility1" | "wheelchair" | "elevation",
    duration_limit: number
  }
}

interface BikeMode {
  max_duration: number
}

interface CarMode {
  max_duration: number
}

interface CarParkingMode {
  max_car_duration: number,
  ppr_search_options: {
    profile: "default",
    duration_limit: number
  }
}
