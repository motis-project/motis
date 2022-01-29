import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import DateTimeService from './services/DateTimeService';
import MOTISPostService from './services/MOTISPostService';
import TranslationService from './services/TranslationService';
import store from './store';
import Interval from './models/SmallTypes/Interval';
import InitialScheduleInfoResponseContent from './models/InitRequestResponseContent';

const app = createApp(App)
app.use(store);
app.use(TranslationService, "de-DE");
app.use(MOTISPostService);

let initTime: Interval = {begin: 0, end: 0};
app.config.globalProperties.$postService.getInitialRequestScheduleInfo().then((resp: InitialScheduleInfoResponseContent) => {
    initTime = {begin: resp.begin, end: resp.end};
});

const interval = setInterval(() => {
    if (TranslationService.service !== null && TranslationService.service.isLoaded && initTime.begin > 0) {
        app.use(router(TranslationService.service));
        const initDate = new Date(initTime.begin * 1000);
        const now = new Date();
        const initialDateTime = new Date(initDate.getFullYear(), initDate.getMonth(), initDate.getDate(), now.getHours(), now.getMinutes(), now.getSeconds(), now.getMilliseconds()).valueOf();
        app.use(DateTimeService, initialDateTime, initTime.end * 1000);
        app.mount('#app');
        clearInterval(interval);
    }
}, 10);
