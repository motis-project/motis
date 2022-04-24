import { App, reactive } from 'vue'
import { DateTime } from 'luxon'
import Interval from '@/models/SmallTypes/Interval';


export class DateTimeService {
  private _dateTime: number;
  public intervalFromServer: Interval;
  private readonly timeFormat: string = "HH:mm";
  private readonly dateFormat: string = "dd.MM.yyyy";
  private readonly simulationTimeFormat: string = "dd.MM.yyyy HH:mm:ss";
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  public mapSetTimeOffset: (offset: number) => void = () => {}

  public constructor(initialDateTime: number, intervalFromServer: Interval) {
    this._dateTime = initialDateTime;
    this.intervalFromServer = intervalFromServer;
  }

  public get dateTime(): number {
    return this._dateTime;
  }

  public set dateTime(value: number) {
    this._dateTime = value;
    this.mapSetTimeOffset(this._dateTime - Date.now().valueOf())
  }

  public get dateTimePure(): number {
    return this._dateTime;
  }

  public set dateTimePure(value: number) {
    this._dateTime = value;
  }

  public get date(): Date {
    return new Date(this._dateTime);
  }

  public get endDate(): string {
    return DateTime.fromMillis(this.intervalFromServer.end).toFormat(this.dateFormat);
  }

  public get dateTimeInSeconds(): number {
    return Math.floor(this._dateTime / 1000);
  }

  public getDateString(dateTime?: number): string {
    if(!dateTime) {
      dateTime = this._dateTime;
    }
    return DateTime.fromMillis(dateTime).toFormat(this.dateFormat);
  }

  public getTimeString(dateTime?: number, seconds?: boolean): string {
    if(!dateTime) {
      dateTime = this._dateTime;
    }
    if(seconds) {
      return DateTime.fromMillis(dateTime).toFormat(this.simulationTimeFormat)
    }
    else {
      return DateTime.fromMillis(dateTime).toFormat(this.timeFormat)
    }
  }

  public parseDate(dateToParse: string): Date {
    const res = DateTime.fromFormat(dateToParse, this.dateFormat).toJSDate();
    const date = this.date;
    return new Date(res.getFullYear(), res.getMonth(), res.getDate(),
      date.getHours(), date.getMinutes(), date.getSeconds(), date.getMilliseconds());
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

class DateTimeServicePlugin {
  public service: DateTimeService = {} as DateTimeService;
  install(app: App, initialDateTime: number, intervalFromServer: Interval) {
    this.service = reactive(new DateTimeService(initialDateTime, intervalFromServer)) as DateTimeService;
    window.setTimeout(() => {
      this.service.dateTimePure += 1000;
      window.setInterval(() => this.service.dateTimePure += 1000, 1000)
    }, 1000 - new Date().getMilliseconds())
    app.config.globalProperties.$ds = this.service;
  }
}

export default new DateTimeServicePlugin();
