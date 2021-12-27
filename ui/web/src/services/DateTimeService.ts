import { App, reactive } from 'vue'
import { DateTime } from 'luxon'
import { TranslationService } from './TranslationService';


class DateTimeService {
  public dateTime: number;
  private readonly timeFormat: string = "HH:mm";
  private _ts: TranslationService;

  public constructor(ts: TranslationService) {
    const now = new Date();
    this.dateTime = new Date(2021, 11, 25, now.getHours(), now.getMinutes(), now.getSeconds(), now.getMilliseconds()).valueOf()
    this._ts = ts;
  }

  public get date(): Date {
    return new Date(this.dateTime);
  }

  public get dateTimeInSeconds(): number {
    return Math.floor(this.dateTime / 1000);
  }

  public getDateString(dateTime?: number): string {
    if(!dateTime) {
      dateTime = this.dateTime;
    }

    return DateTime.fromMillis(dateTime).toFormat(this._ts.t.dateFormat);
  }

  public getTimeString(dateTime?: number): string {
    if(!dateTime) {
      dateTime = this.dateTime;
    }
    return DateTime.fromMillis(dateTime).toFormat(this.timeFormat)
  }

  public parseDate(dateToParse: string): Date {
    return DateTime.fromFormat(dateToParse, this._ts.t.dateFormat).toJSDate();
  }

  public parseTime(timeToParse: string) : Date {
    const d = this.date;
    const t = DateTime.fromFormat(timeToParse, this.timeFormat).toJSDate();
    return new Date(d.getFullYear(), d.getMonth(), d.getDate(), t.getHours(), t.getMinutes());
  }

}






declare module '@vue/runtime-core' {
  interface ComponentCustomProperties {
    $ds: DateTimeService,
  }
}

export default {
  install: (app: App): void => {
    const service = reactive(new DateTimeService(app.config.globalProperties.$ts));
    window.setTimeout(() => { service.dateTime += 1000; window.setInterval(() => service.dateTime += 1000, 1000) }, 1000 - new Date().getMilliseconds())
    app.config.globalProperties.$ds = service;
  }
}
