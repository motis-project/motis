import { App } from "vue";

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

// eslint-disable-next-line @typescript-eslint/no-empty-function
const defaultDelegateValue = (): void => {}

export class MotisMapService {
  public initialized = false;

  //assigned in map
  public mapInit: (id: string) => void = defaultDelegateValue;
  public mapFlyTo: (options: MapFlyToOptions) => void = defaultDelegateValue;
  public mapFitBounds: (options: MapFitBoundsOptions) => void = defaultDelegateValue;

  //called in map
  public mapShowContextMenu: (mapPostion: MapClickInfo) => void = defaultDelegateValue;
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
