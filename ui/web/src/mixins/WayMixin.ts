import { TranslationService } from "../services/TranslationService";
import { Router } from "vue-router";
import Stop from "../models/Stop";

export default {
  methods: {
    getPastOrFuture(currentDate: Date, timeInSeconds: number): string {
      const date = new Date(timeInSeconds * 1000);
      return date < currentDate ? 'past' : 'future';
    },
    goToStop(router: Router, stop: Stop): void {
      router.push({
        name: "StationTimetable",
        params: {
          id: stop.station.id
        },
      });
    },
    getReadableDuration(timeInSecondsDeparture: number, timeInSecondsArrival: number, ts: TranslationService): string {
      const time = new Date(
        (timeInSecondsArrival - timeInSecondsDeparture) * 1000
      );
      let res = "";
      if (time.getDate() > 1) {
        res += ts.formatTranslate("days", (time.getDate() - 1).toString());
      }
      if (res !== "") {
        res += " ";
      }
      if (time.getHours() > 1) {
        res += ts.formatTranslate("hours", (time.getHours() - 1).toString());
      }
      if (res !== "") {
        res += " ";
      }
      if (time.getMinutes() > 0) {
        res += ts.formatTranslate("minutes", time.getMinutes().toString());
      }
      return res;
    },
    getProgress(stop: Stop, nextStop: Stop, currentDateTimeInSeconds: number): number {
      const diff = nextStop.arrival.time - stop.departure.time;
      const diffWithCurrent = nextStop.arrival.time - currentDateTimeInSeconds;
      if(diffWithCurrent < 0) {
        return 100;
      }
      else if(diffWithCurrent > diff) {
        return 0;
      }
      else {
        return 100 - (diffWithCurrent / diff) * 100;
      }
    },
  }
}
