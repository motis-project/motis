import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import ConnectionSearch from "../views/ConnectionSearch.vue"
import TrainSearch from "../views/TrainSearch.vue"

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
      subOverlay: TrainSearch
    }
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
