import StationGuess from "./StationGuess";
import TransportTrip from "./TransportTrip";

export default interface Event {
  trips: TransportTrip[],
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

