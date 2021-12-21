export default interface StationGuess {
    id: string,
    name: string,
    pos: {
        lat: number,
        lng: number
    }
}

export interface StationGuessResponseContent {
    guesses: StationGuess[]
}
