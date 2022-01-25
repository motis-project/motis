import DateTimeService from "../../src/services/DateTimeService";
import TranslationService from "../../src/services/TranslationService";
import { config } from "@vue/test-utils";
import axios from "axios";


const mockedAxios = axios as jest.Mocked<typeof axios>

export function prepareGlobals(done: () => void) : void {
    const ts = TranslationService;
    mockedAxios.get.mockResolvedValueOnce({data: {}})
    ts.createTranslationService("de-DE");
    const interval = setInterval(() => {
      if (ts.service !== null && ts.service.isLoaded) {
        const now = new Date();
        const initialDateTime = new Date(2021, 11, 25, now.getHours(), now.getMinutes(), now.getSeconds(), now.getMilliseconds()).valueOf()
        config.global.plugins = [[ts, "de-DE"], [DateTimeService, initialDateTime]]
        clearInterval(interval);
        done();
      }
    }, 10);
}

export function getRandomInt(min: number, max: number): number {
  max++;
  return Math.floor(Math.random() * (max - min)) + min;
}
