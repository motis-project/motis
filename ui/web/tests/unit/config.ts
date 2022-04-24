import DateTimeService from "../../src/services/DateTimeService";
import TranslationService from "../../src/services/TranslationService";
import { config } from "@vue/test-utils";
import axios from "axios";
import MOTISPostService from "../../src/services/MOTISPostService";
import MOTISMapServicePlugin from "../../src/services/MOTISMapService"

jest.mock("axios");

const mockedAxios = axios as jest.Mocked<typeof axios>

const ts = TranslationService;
mockedAxios.get.mockResolvedValueOnce({data: {
  parking: "Parking"
}})
ts.createTranslationService("de-DE");

const int = {begin: 1640386800, end: 1640646000};
config.global.plugins = [[ts, "de-DE"], [DateTimeService, int.begin * 1000, int], MOTISPostService, MOTISMapServicePlugin]
