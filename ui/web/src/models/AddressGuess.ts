export default interface AddressGuess {

    pos: {
        lat: number,
        lng: number
    },
    name: string,
    type: string,
    regions: Region[] 

}
export interface AddressGuessResponseContent{
    guesses: AddressGuess[]
}
interface Region {
    name: string,
    admin_level: number
}