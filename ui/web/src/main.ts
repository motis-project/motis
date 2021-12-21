import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import DateTimeService from './services/DateTimeService';
import MOTISPostService from './services/MOTISPostService';
import TranslationService from './services/TranslationService';

const app = createApp(App)
app.use(TranslationService, "de-DE");
const interval = setInterval(() => {
    if (TranslationService.service.isLoaded) {
        TranslationService.service.isLoaded;
        app.use(router(TranslationService.service));
        app.use(MOTISPostService);
        app.use(DateTimeService);
        app.mount('#app');
        clearInterval(interval);
    }
}, 10);

