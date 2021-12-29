import { Connection, Position, Station } from './ConnectionTypes'
import { SearchOptions } from './PPRTypes';
import { Interval } from './RoutingTypes';

export interface IntermodalRoutingRequest {
    start_type: string,
    start: IntermodalPretripStartInfo | PretripStartInfo,
    startModes: [FootModeInfo | BikeModeInfo | CarModeInfo | FootPPRInfo | CarParkingModeInfo],
    destination: Station | Position,
    destinationModes: [FootModeInfo | BikeModeInfo | CarModeInfo | FootPPRInfo | CarParkingModeInfo],
    searchType: 'DeafaultSearch' | 'SingleCriterion' | 'SingleCriterionNoIntercity' | 'LateConnections' | 'LateConnectionsTest' | 'Accessibility',
    searchDir: 'Forward' | 'Backward'
}


export interface IntermodalRoutingResponse {
    content_type: string,
    content: {
        connections: [Connection],
        interval_begin: number,
        interval_end: number,
        direct_connections: [Connection]
    },
    id: number
}


export interface IntermodalPretripStartInfo {
    position: Position,
    interval: Interval,
    minConnectionCount: number,
    extendIntervalEarlier: boolean,
    extendIntervalLater: boolean
}


export interface PretripStartInfo {
    station: Station,
    interval: Interval,
    minConectionCount: number,
    extendIntervalEarlier: boolean,
    extendIntervalLater: boolean
}


interface FootModeInfo {
    maxDuration: number
}


interface BikeModeInfo {
    maxDuration: number
}


interface CarModeInfo {
    maxDuration: number
}


interface FootPPRInfo {
    searchOptions: SearchOptions
}


interface CarParkingModeInfo {
    maxCarDuration: number,
    pprSearchOptions: SearchOptions
}