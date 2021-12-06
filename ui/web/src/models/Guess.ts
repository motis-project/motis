export interface Guess {
    id: string,
    name: string,
    pos: {
        lat: number,
        lng: number
    }
}

export default interface GuessResponseContent {
    guesses: Guess[]
}