/* eslint-disable camelcase */

export default interface Trip {
  station_id: string,
  train_nr: number,
  time: number,
  target_station_id: string,
  target_time: number,
  line_id: string
}
