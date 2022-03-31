/* eslint-disable camelcase */
import StationGuess from "./StationGuess";

export default interface Stop {
  station: StationGuess,
  arrival: ScheduleEntry,
  departure: ScheduleEntry,
  exit: boolean,
  enter: boolean
}

interface ScheduleEntry {
  time: number,
  schedule_time: number,
  track: string,
  schedule_track: string,
  valid: boolean,
  reason: string
}
