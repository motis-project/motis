/* eslint-disable camelcase */
import Range from "./SmallTypes/Range";

export default interface Transport {
  range: Range
  category_name: string,
  category_id: number,
  clasz: number,
  train_nr: number,
  line_id: string,
  name: string,
  provider: string,
  direction: string
}
