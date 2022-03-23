import { Connection, Position, Station } from './Connection'
import { SearchOptions } from './PPRTypes';
import { Interval } from './RoutingTypes';
import { ScheduleInfoResponse } from './ScheduleInfo';

export interface IntermodalRoutingRequest {
    start_type: string,
    start: IntermodalPretripStartInfo | PretripStartInfo,
    start_modes: Mode[],
    destination: Station | Position,
    destination_modes: Mode[],
    search_type: 'DefaultSearch' | 'SingleCriterion' | 'SingleCriterionNoIntercity' | 'LateConnections' | 'LateConnectionsTest' | 'Accessibility',
    search_dir: 'Forward' | 'Backward'
}


export interface IntermodalPretripStartInfo {
    position: Position,
    interval: Interval,
    min_connection_count: number,
    extend_interval_earlier: boolean,
    extend_interval_later: boolean
}


export interface PretripStartInfo {
    station: Station,
    interval: Interval,
    min_conection_count: number,
    extend_interval_earlier: boolean,
    extend_interval_later: boolean
}


export interface Mode {
    mode_type: string,
    mode: FootModeInfo | BikeModeInfo | CarModeInfo | FootPPRInfo | CarParkingModeInfo
}


interface FootModeInfo {
    max_duration: number
}


interface BikeModeInfo {
    max_duration: number
}


interface CarModeInfo {
    max_duration: number
}


export interface FootPPRInfo {
    search_options: SearchOptions
}


interface CarParkingModeInfo {
    max_car_duration: number,
    ppr_search_options: SearchOptions
}


export interface IntermodalRoutingResponse {
    content_type: string,
    content: {
        connections: Connection[],
        interval_begin: number,
        interval_end: number,
        direct_connections: Connection[]
        statistics: Statistic[]
    },
    destination: {
        target: string,
        type: string
    }
    id: number
}


interface IntermodalRoutingR {
    connections: Connection[],
    interval_begin: number,
    interval_end: number,
    direct_connections: Connection[]
    statistics: Statistic[]
}


export interface elmAPIResponse {
    content: ScheduleInfoResponse | IntermodalRoutingR
    content_type: string,
    destination: {
        target: string,
        type: string
    }
    id : number
}


interface Statistic {
    category: string,
    entries: Entry[]
}


interface Entry {
    name: string,
    value: number
}