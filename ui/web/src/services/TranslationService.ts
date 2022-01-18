import axios from 'axios';
import { App, reactive } from 'vue'

/* eslint-disable camelcase */
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
  searchProfile_defalt: string,
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
  useParking: string,
  parking: string
}
/* eslint-enable camelcase */

export class TranslationService {
  private _t: Translation = reactive({}) as Translation;
  public isLoaded = false;
  public availableLocales = [
    "de-DE",
    "en-US"
  ];

  public get t(): Translation {
    return this._t;
  }
  public currentLocale = "";

  public constructor(locale: string) {
    this.changeLocale(locale);
  }

  public changeLocale(locale: string): void {
    this.currentLocale = locale;
    this.loadLocale(locale).then(t => {
      Object.assign(this._t, t);
      this.isLoaded = true;
    });
  }

  public countTranslate(str: string, count: number): string {
    const arr = this.t[str];
    if (!arr || !(Array.isArray(arr))) {
      return '';
    }
    const index = count > arr.length - 1 ? arr.length - 1 : count;
    return this.format(arr[index], count.toString());
  }

  public formatTranslate(str: string, ...formatOptions: string[]): string {
    return this.format(this.t[str] as string, ...formatOptions)
  }

  private loadLocale(locale: string): Promise<Translation> {
    return axios.get("locales/" + locale + ".json").then(r => r.data).then(json => json as Translation);
  }

  private format(str: string, ...formatOptions: string[]) {
    if (!str) {
      return '';
    }

    return str.replace(/{(\d+)}/g, function (match, number) {
      return typeof formatOptions[number] !== 'undefined'
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

class TranslationServicePlugin {
  public service: TranslationService | null = null;
  public install(app: App, options: string) {
    this.createTranslationService(options);
    if(this.service !== null) {
      app.config.globalProperties.$ts = this.service;
      app.config.globalProperties.$t = this.service.t;
    }
  }
  public createTranslationService(options: string) {
    if(this.service === null) {
      this.service = reactive(new TranslationService(options)) as TranslationService;
    }
  }
}

export default new TranslationServicePlugin();
