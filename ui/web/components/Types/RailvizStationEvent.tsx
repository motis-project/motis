import { TransportInfo, Station } from "./Connection"

export interface RailVizStationResponse {
    content_type: string,
    content: StationEvents,
    destination: { type: string, target: string },
    id: number
}


export interface RailVizTripGuessResponse {
    content_type: string,
    content: {trips: {first_station: Station, trip_info: Trip}[]}
    destination: { type: string, target: string },
    id: number
}


export interface StationEvents{
    station: Station,
    events: Events[]
}

export interface Events{
    trips: Trip[],
    type: string,
    event: {
        time: number,
        schedule_time: number,
        track: string,
        schedule_track: string,
        valid: boolean,
        reason: string
    },
    dummyEvent?: string
}

export interface Trip{
    id: {
        station_id: string,
        train_nr: number,
        time: number,
        target_station_id: string,
        target_time: number,
        line_id: string
    },
    transport: TransportInfo
}