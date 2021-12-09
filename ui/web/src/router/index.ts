import { createRouter, createWebHashHistory, RouteRecordRaw } from 'vue-router'
import ConnectionSearch from "../views/ConnectionSearch.vue"
import TrainSearch from "../views/TrainSearch.vue"
import Trip from '../views/Trip.vue'
import {Router, RouteLocationNormalizedLoaded} from "vue-router"

export const SubOverlayNames = ["TrainSearch", "Trip"]

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
  },
  {
    path: '/trip/:station_id/:train_nr/:time/:target_station_id/:target_time/:line_id',
    name: "Trip",
    components: {
      overlay: ConnectionSearch,
      subOverlay: Trip
    },
    props: {
      subOverlay: route => ({
        trip: {
          station_id: route.params.station_id,
          train_nr: route.params.train_nr,
          time: route.params.time,
          target_station_id: route.params.target_station_id,
          target_time: route.params.target_time,
          line_id: route.params.line_id
        }
      })
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
