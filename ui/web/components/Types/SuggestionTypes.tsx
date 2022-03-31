import { Position, Station } from "./Connection";

export interface AddressSuggestionResponse {
    content:
    {
        guesses: Address[]
    },
    content_type: string,
    destination: {
        target: string,
        type: string
    },
    id : number
}


export interface StationSuggestionResponse {
    content:
    {
        guesses: Station[]
    },
    content_type: string,
    destination: {
        target: string,
        type: string
    },
    id : number
}


export interface Address {
    pos: Position,
    name: string,
    type_: string,
    regions: Region[]
}


export interface Region {
    name: string,
    admin_level: number
}