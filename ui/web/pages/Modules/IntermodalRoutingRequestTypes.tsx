import { Position, Station } from './ConnectionTypes'
import { SearchOptions } from './PPRTypes';
import { Interval, SearchDirection, SearchType } from './RoutingTypes';

export interface IntermodalRoutingRequest {
    start_type: string,
    start: IntermodalPretripStartInfo | PretripStartInfo,
    startModes: [FootModeInfo | BikeModeInfo | CarModeInfo | FootPPRInfo | CarParkingModeInfo],
    destination: Station | Position,
    destinationModes: [FootModeInfo | BikeModeInfo | CarModeInfo | FootPPRInfo | CarParkingModeInfo],
    searchType: SearchType,
    searchDir: SearchDirection
}


interface IntermodalPretripStartInfo {
    position: Position,
    interval: Interval,
    minConnectionCount: number,
    extendIntervalEarlier: boolean,
    extendIntervalLater: boolean
}


interface PretripStartInfo {
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