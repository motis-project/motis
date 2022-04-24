/* eslint-disable camelcase */
import Range from "./SmallTypes/Range";

export default interface CustomMovement {
  range: Range,
  mumo_id: number,
  price: number,
  accessibility: number,
  mumo_type: "car" | "bike" | "foot" | ""
}
