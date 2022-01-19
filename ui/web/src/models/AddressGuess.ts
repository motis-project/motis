/* eslint-disable camelcase */
import Position from "./SmallTypes/Position";

export default interface AddressGuess {
  pos: Position,
  name: string,
  type: string,
  regions: Region[]

}
export interface AddressGuessResponseContent {
  guesses: AddressGuess[]
}
interface Region {
  name: string,
  admin_level: number
}
