import { App } from "vue";
import Position from "../models/SmallTypes/Position"
import StationGuess from "../models/StationGuess";
import Trip from "../models/Trip";

interface MapClickInfo {
  mouseX: number,
  mouseY: number,
  lat: number,
  lng: number,
}

interface MapFlyToOptions {
  mapId: string,
  lng: number,
  lat: number,
  zoom?: number,
  bearing?: number,
  pitch?: number,
  animate: boolean
}

interface MapFitBoundsOptions {
  mapId: string,
  coords: number[][]
}

interface MapSetMarkersOptions {
  startPosition?: Position,
  destinationPosition?: Position,
  startName?: string,
  destinationName?: string
}

interface RVDeatilFilterOptions {
  trains: RVConnectionTrain[],
  walks: RVConnectionWalkOptions[],
  interchangeStations: StationGuess[]
}

interface RVConnectionTrain{
  sections: RVConnectionSection[],
  trip?: Trip
}

interface RVConnectionSection {
  departureStation: StationGuess,
  arrivalStation: StationGuess,
  scheduledDepartureTime: Date, // Time in elm equal to Date in ts???
  scheduledArrivalTime: Date
}

interface RVConnectionWalkOptions {
  departureStation: StationGuess,
  arrivalStation: StationGuess,
  polyline?: number[],
  mumoType: string,
  duration: number,
  accessibility: number
}

interface RVConnectionsOptions {
  mapId: string,
  connections: RVConnection[],
  lowestId: number
}

interface RVConnection {
  id: number,
  stations: StationGuess[],
  trains: RVConnectionTrain[],
  walks: RVConnectionWalkOptions[]
}

interface MapLocaleOptions {
  start: string,
  destination: string
}

interface MapTooltipOptions {
  mouseX: number,
  mouseY: number,
  hoveredTrain?: RVTrain,
  hoveredStation?: string,
  hoveredTripSegments?: RVConnectionSegmentTrip[],
  hoveredWalkSegment?: RVConnectionSegmentWalk
}

interface RVTrain {
  names: string[],
  departureTime: Date,
  arrivalTime: Date,
  scheduledDepartureTime: Date,
  scheduledArrivalTime: Date,
  hasDepartureDelayInfo: boolean,
  hasArrivalDelayInfo: boolean,
  departureStation: string,
  arrivalStation: string
}

interface RVConnectionSegmentTrip {
  connectionIds: number[],
  trip: Trip[],
  // eslint-disable-next-line camelcase
  d_station_id: string,
  // eslint-disable-next-line camelcase
  a_station_id: string
}

interface RVConnectionSegmentWalk {
  connectionIds: number,
  walk : RVConnectionWalkOptions
}

interface MapInfoOptions {
  scale: number,
  zoom: number,
  pixelBounds: MapPixelBounds,
  geoBounds: MapGeoBounds,
  railVizBounds: MapGeoBounds,
  center: Position,
  bearing: number,
  pitch: number,
}

interface MapPixelBounds {
  north: number,
  west: number,
  width: number,
  height: number
}

interface MapGeoBounds {
  north: number,
  west: number,
  south: number,
  east: number
}

// eslint-disable-next-line @typescript-eslint/no-empty-function
const defaultDelegateValue = (): void => {}

export class MotisMapService {
  public initialized = false;

  //assigned in map
  public mapInit: (id: string) => void = defaultDelegateValue;
  public mapFlyTo: (options: MapFlyToOptions) => void = defaultDelegateValue;
  public mapFitBounds: (options: MapFitBoundsOptions) => void = defaultDelegateValue;
  public mapSetMarkers: (options: MapSetMarkersOptions) => void = defaultDelegateValue;
  public mapSetDetailFilter: (options: RVDeatilFilterOptions) => void = defaultDelegateValue;
  public mapUpdateWalks: (options: RVConnectionWalkOptions) => void = defaultDelegateValue;
  public mapSetConnections: (options: RVConnectionsOptions) => void = defaultDelegateValue;
  public mapHighlightConnections: (options: number[]) => void = defaultDelegateValue;
  public setTimeOffset: (options: number) => void = defaultDelegateValue;
  // eslint-disable-next-line @typescript-eslint/ban-types
  public setPPRSearchOptions: (options: object) => void = defaultDelegateValue; // JSON.Encode.Value equals to object in ts???
  public mapSetLocale: (options: MapLocaleOptions) => void = defaultDelegateValue;
  // public localStorageSet: (options: StringTuples) => void = defaultDelegateValue;
  // eslint-disable-next-line @typescript-eslint/ban-types
  public handleRailVizError: (options: object) => void = defaultDelegateValue;
  public clearRailVizError: () => void = defaultDelegateValue;
  public showTripDetails: (options: Trip) => void = defaultDelegateValue;
  public showStationDetails: (options: string) => void = defaultDelegateValue;
  public mapUpdate: (options: MapInfoOptions) => void = defaultDelegateValue;


  //called in map
  public mapShowContextMenu: (mapPostion: MapClickInfo) => void = defaultDelegateValue;
  public mapUseTrainClassColors: (options: boolean) => void = defaultDelegateValue;
  public mapShowTrains: (options: boolean) => void = defaultDelegateValue;
  public mapCloseContextMenu: () => void = defaultDelegateValue;
  public mapSetTooltip: (options: MapTooltipOptions) => void = defaultDelegateValue;
}




declare module '@vue/runtime-core' {
  interface ComponentCustomProperties {
    $mapService: MotisMapService
  }
}

class MapServicePlugin {
  public service: MotisMapService = {} as MotisMapService;
  public install(app: App) {
    this.service = new MotisMapService();
    app.config.globalProperties.$mapService = this.service;
  }
}

export default new MapServicePlugin();
