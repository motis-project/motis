import { App, reactive } from 'vue'

interface Translation {
  [key: string]: string | string[]

  dateFormat: string,
  start: string,
  destination: string,
  date: string,
  time: string,
  departure: string,
  arrival: string,
  startTransports : string,
  destinationTransports : string,
  walk : string,
  bike: string,
  car: string,
  maxDuration: string,
  searchProfile_default: string,
  searchProfile_accessibility: string,
  searchProfile_wheelchair: string,
  searchProfile_elevation: string,
  trainNr: string,
  provider : string,
  changes: string[],
  min: string,
  hours: string,
  days: string,
  stop: string[],
  earlier : string,
  later: string,
  profile: string,
  useParking: string
}

class TranslationService {
  private _t: Translation = reactive({}) as Translation;

  public get t(): Translation {
    return this._t;
  }
  public currentLocale: string;

  public constructor(locale: string) {
    this.currentLocale = locale;
    this.changeLocale(locale);
  }

  public changeLocale(locale: string) {
    this.currentLocale = locale;
    this.loadLocale(locale).then(t => {
      Object.assign(this._t, t);
    });
  }

  public countTranslate(str: string, count: number): string {
    let arr = this.t[str];
    if (!arr || !(Array.isArray(arr))) {
      return '';
    }
    let index = count > arr.length - 1 ? arr.length - 1 : count;
    return this.format(arr[index], count.toString());
  }

  public formatTranslate(str: string, ...formatOptions: string[]): string {
    return this.format(this.t[str] as string, ...formatOptions)
  }

  private loadLocale(locale: string): Promise<Translation> {
    return fetch("locales/" + locale + ".json").then(t => t.json()).then(json => json as Translation);
  }

  private format(str: string, ...formatOptions: string[]) {
    if (!str) {
      return '';
    }

    return str.replace(/{(\d+)}/g, function (match, number) {
      return typeof formatOptions[number] != 'undefined'
        ? formatOptions[number]
        : match;
    })
  }
}







declare module '@vue/runtime-core' {
  interface ComponentCustomProperties {
    $t: Translation,
    $ts: TranslationService,
  }
}

export default {
  install: (app: App, options: string) => {
    let service = reactive(new TranslationService(options));
    app.config.globalProperties.$ts = service;
    app.config.globalProperties.$t = service.t;
  }
}