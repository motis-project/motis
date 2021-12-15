import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import MOTISPostService from './services/MOTISPostService';
import TranslationService from './services/TranslationService';

var app = createApp(App)
app.use(TranslationService, "de-DE");
app.use(router)
app.use(MOTISPostService);

app.mount('#app')
