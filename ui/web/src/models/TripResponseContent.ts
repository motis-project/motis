import Attribute from "./SmallTypes/Attribute";
import FreeText from "./SmallTypes/FreeText";
import Problem from "./SmallTypes/Problem";
import Stop from "./Stop";
import Transport from "./Transport";
import Trip from "./Trip";

export default interface TripResponseContent {
  stops: Stop[],
  transports: {
    move: Transport
  }[],
  trips: {
    id: Trip
  }[],
  attributes: Attribute[],
  free_texts: FreeText[],
  problems: Problem[],
  night_penalty: number,
  db_costs: number,
  status: string
}