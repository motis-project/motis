import { App } from "vue";

interface MapPostion {
  mouseX: number,
  mouseY: number,
  lat: number,
  lng: number,
}

interface MapFlyToOptions {
  mapId: number,
  animate: boolean,
  lng: number,
  lat: number,
  zoom?: boolean,
  pitch?: number,
  bearing?: number,
}

interface MapFitBoundsOptions {
  mapId: number,
  coords: number[] //FIX
}

interface MapInitDelegate {
  (id: string): void
}

interface MapShowContextMenuDelegate {
  (mapPostion: MapPostion): void
}

interface MapFlyToDelegate {
  (options: MapFlyToOptions): void
}

interface MapFitBoundsDelegate {
  (options: MapFitBoundsOptions): void
}

interface VoidDelegate {
  (): void
}

// eslint-disable-next-line @typescript-eslint/no-empty-function
const defaultDelegateValue = (): void => {}

export class MotisMapService {
  public initialized = false;

  //assigned in map
  public mapInit: MapInitDelegate = defaultDelegateValue;
  public mapFlyTo: MapFlyToDelegate = defaultDelegateValue;
  public mapFitBounds: MapFitBoundsDelegate = defaultDelegateValue;

  //called in map
  public mapShowContextMenu: MapShowContextMenuDelegate = defaultDelegateValue;
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
