export default interface Stop {
  station: {
    id: string,
    name: string,
    pos: {
      lat: number,
      lng: number
    }
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