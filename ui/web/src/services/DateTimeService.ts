import { App, reactive } from 'vue'
import { DateTime } from 'luxon'


export class DateTimeService {
  public dateTime: number;
  private readonly timeFormat: string = "HH:mm";
  private readonly dateFormat: string = "dd.MM.yyyy";

  public constructor(initialDateTime: number) {
    this.dateTime = initialDateTime;
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

    return DateTime.fromMillis(dateTime).toFormat(this.dateFormat);
  }

  public getTimeString(dateTime?: number): string {
    if(!dateTime) {
      dateTime = this.dateTime;
    }
    return DateTime.fromMillis(dateTime).toFormat(this.timeFormat)
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

export default {
  install: (app: App, initialDateTime: number): void => {
    const service = reactive(new DateTimeService(initialDateTime));
    window.setTimeout(() => { service.dateTime += 1000; window.setInterval(() => service.dateTime += 1000, 1000) }, 1000 - new Date().getMilliseconds())
    app.config.globalProperties.$ds = service;
  }
}
