import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router'
import ConnectionSearch from "../views/ConnectionSearch.vue"
import TrainSearch from "../views/TrainSearch.vue"
import {Router, RouteLocationNormalizedLoaded} from "vue-router"

export const SubOverlayNames = ["TrainSearch"]

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'ConnectionSearch',
    components : {
      overlay: ConnectionSearch
    }
  },
  {
    path: '/trips',
    name: "TrainSearch",
    components: {
      overlay: ConnectionSearch,
      subOverlay: TrainSearch
    }
  }
]

const router = createRouter({
  history: createWebHashHistory(process.env.BASE_URL),
  routes
})

declare module '@vue/runtime-core' {
  interface ComponentCustomProperties {
      $router: Router,
      $route: RouteLocationNormalizedLoaded
  }
}

export default router
