import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import DateTimeService from './services/DateTimeService';
import MOTISPostService from './services/MOTISPostService';
import TranslationService from './services/TranslationService';

var app = createApp(App)
app.use(TranslationService, "de-DE");
let interval = setInterval(() => {
    if (TranslationService.service.isLoaded) {
        TranslationService.service.isLoaded;
        app.use(router);
        app.use(MOTISPostService);
        app.use(DateTimeService);
        app.mount('#app');
        clearInterval(interval);
    }
}, 10);

