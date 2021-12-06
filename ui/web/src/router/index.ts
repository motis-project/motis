import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import ConnectionSearch from "../views/ConnectionSearch.vue"
import TrainSearch from "../views/TrainSearch.vue"
import {Router} from "vue-router"

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    components : {
      overlay: ConnectionSearch
    }
  },
  {
    path: '/trips',
    components : {
      overlay: ConnectionSearch,
      subOverlay: TrainSearch
    }
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

declare module '@vue/runtime-core' {
  interface ComponentCustomProperties {
      $router: Router
  }
}

export default router
