import Position from "./SmallTypes/Position";

export default interface StationGuess {
  id: string,
  name: string,
  pos: Position
}

export interface StationGuessResponseContent {
  guesses: StationGuess[]
}
