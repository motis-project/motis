/* eslint-disable camelcase */
import Position from "./SmallTypes/Position";

export default interface Stop {
  station: {
    id: string,
    name: string,
    pos: Position
  },
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
