import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

var app = createApp(App)
app.use(router)


app.mount('#app')
