import { createApp } from 'vue'
import AppComponent from './App.vue'
import router from './router'
import DateTimeService from './services/DateTimeService';
import MOTISPostService from './services/MOTISPostService';
import TranslationService from './services/TranslationService';
import store from './store';
import Interval from './models/SmallTypes/Interval';
import InitialScheduleInfoResponseContent from './models/InitRequestResponseContent';
import MOTISMapServicePlugin, { MotisMapService } from './services/MOTISMapService';

const app = createApp(AppComponent)
app.use(store);
app.use(TranslationService, "de-DE");
app.use(MOTISPostService);
app.use(MOTISMapServicePlugin);

let intervalFromServer: Interval = { begin: 0, end: 0 };
app.config.globalProperties.$postService.getInitialRequestScheduleInfo().then((resp: InitialScheduleInfoResponseContent) => {
  intervalFromServer = { begin: resp.begin, end: resp.end };
});

const interval = setInterval(() => {
  if (TranslationService.service !== null && TranslationService.service.isLoaded && intervalFromServer.begin > 0) {
    app.use(router(TranslationService.service));
    const initDate = new Date(intervalFromServer.begin * 1000);
    const now = new Date();
    const initialDateTime = new Date(initDate.getFullYear(), initDate.getMonth(), initDate.getDate(), now.getHours(), now.getMinutes(), now.getSeconds(), now.getMilliseconds()).valueOf();
    app.use(DateTimeService, initialDateTime, intervalFromServer);
    app.mount('#app');
    clearInterval(interval);
  }
}, 10);

declare global {
  interface Window {
    mapService: MotisMapService
  }
}
window.mapService = MOTISMapServicePlugin.service;

const intervalMap = setInterval(() => {
  if (MOTISMapServicePlugin.service.initialized) {
    MOTISMapServicePlugin.service.mapInit("map");
    clearInterval(intervalMap);
  }
}, 10);
