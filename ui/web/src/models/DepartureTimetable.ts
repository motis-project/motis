import StationGuess from "./StationGuess";
import Transport from "./Transport";
import Trip from "./Trip";

export default interface DepatrureTimetable {
  id: Trip,
  transport: Transport
}

export interface Event {
  trips: DepatrureTimetable[],
  type: string,
  event: {
    time: number,
    schedule_time: number,
    track: string,
    schedule_track: string,
    valid: Boolean,
    reason: string
  }
}


export interface RailVizStationResponseContent {
  station: StationGuess,
  events: Event[]
}

