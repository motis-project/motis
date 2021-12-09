import Range from "./SmallTypes/Range";

export default interface Trip {
  range: Range
  station_id: string,
  train_nr: number,
  time: number,
  target_station_id: string,
  target_time: number,
  line_id: string
}