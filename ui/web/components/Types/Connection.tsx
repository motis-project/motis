export interface Connection {
    stops: Stop[],
    transports: Transport[],
    trips: Trip[],
    problems: Problem[],
    dummyDay?: string,
    new: string
}

export interface TripViewConnection {
    content: Connection,
    destination: { type: string, target: string },
    id: number
}

export interface FootRouting {
    content: Routess,
    content_type: string,
    destination: { type: string, target: string },
    id: number
}

interface Routess {
    routes: Routes[]
}

interface Routes {
    routes: Route[]
}

interface Route {
    distance: number,
    duration: number,
    duration_exact: number,
    duration_division: number,
    accessibility: number,
    accessibility_exact: number,
    accessibility_division: number,
    elevation_up: number,
    elevation_down: number,
    start: {lat: number, lng: number},
    destination: {lat: number, lng: number},
    steps: Step[],
    path: number[]
}

interface Step {
    step_type: string,
    street_name: string,
    street_type: string,
    crossing_type: string,
    distance: number,
    duration: number,
    accessibility: number,
    path: number[],
    elevation_up: number,
    elevation_down: number,
    incline_up: boolean,
    handrail: string
}

export interface Stop {
    station: Station,
    arrival: EventInfo,
    departure: EventInfo,
    exit: boolean,
    enter: boolean
}


export interface Station {
    id: string,
    name: string,
    pos?: Position
}


export interface Position {
    lat: number,
    lng: number
}


interface EventInfo {
    time?: number
    schedule_time?: number
    track: string,
    reason: 'Schedule' | 'Is' | 'Propagation' | 'Forecast'
}

export interface Transport {
    move: (TransportInfo | WalkInfo)
    move_type: string
}

export interface TransportInfo {
    category_id: number,
    range: Range,
    category_name: string,
    clasz: number,
    train_nr?: number,
    line_id: string,
    name: string,
    provider: string,
    direction: string
}


export interface WalkInfo {
    range: Range,
    mumo_id: number,
    price: number,
    accessibility: number,
    mumo_type: string
}


interface Attribute {
    range: Range,
    code: string,
    text: string
}


interface Range {
    from: number,
    to: number
}


export interface TripId {
    station_id: string,
    train_nr: number,
    time: number,
    target_station_id: string,
    target_time: number,
    line_id: string
}


export interface Trip {
    range: Range,
    id: TripId
}


interface Problem {
    range: Range,
    typ: 'NoProblem' | 'InterchangeTimeViolated' | 'CanceledTrain'
}