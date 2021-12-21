import { createRouter, createWebHashHistory, RouteRecordRaw, Router, RouteLocationNormalizedLoaded } from 'vue-router'
import ConnectionSearch from "../views/ConnectionSearch.vue"
import TrainSearch from "../views/TrainSearch.vue"
import Trip from '../views/Trip.vue'
import StationTimetable from "../views/StationTimetable.vue"
import { TranslationService } from '../services/TranslationService'
import PageNotFound from '../views/PageNotFound.vue'

export const SubOverlayNames = ["TrainSearch", "StationTimetable", "Trip", "StationTimeTableFromTrainSearch"]

/*eslint-disable camelcase*/
const routes: Array<RouteRecordRaw> = [
  {
    path: '/:locale?/',
    name: 'ConnectionSearch',
    components : {
      overlay: ConnectionSearch
    }
  },
  {
    path: '/:locale?/trips',
    name: "TrainSearch",
    components: {
      overlay: ConnectionSearch,
      subOverlay: TrainSearch
    }
  },
  {
    path: '/:locale?/station/:id',
    name: "StationTimetable",
    components: {
      overlay: ConnectionSearch,
      subOverlay: StationTimetable
    },
    props: {
      subOverlay: route => ({
        stationGuess: {
          id: route.params.id,
        }
      })
    }
  },
  {
    path: '/:locale?/trip/:station_id/:train_nr/:time/:target_station_id/:target_time/:line_id?',
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
    },
  },
  {
    path: '/:locale?/station/:id/:time',
    name: 'StationTimeTableFromTrainSearch',
    components: {
      overlay: ConnectionSearch,
      subOverlay: StationTimetable
    },
    props: {
      subOverlay: route => ({
        stationGuess: {
          name: route.params.name,
          id: route.params.id
        },
        tripIdGuess: {
          time: route.params.time
        }
      })
    }
  },
  {
    path: '/:pathMatch(.*)',
    name: 'PageNotFound',
    component: PageNotFound
  }
]
/* eslint-enable camelcase*/

const router = createRouter({
  history: createWebHashHistory(process.env.BASE_URL),
  routes
})

let ts : TranslationService;

router.beforeEach((to, from, next) => {
  let isNextCalled = false;
  let isLocaleInvalid = false;

  if(to.params.locale !== ts.currentLocale) {
    if(ts.availableLocales.includes(to.params.locale as string)) {
      ts.changeLocale(to.params.locale as string);
    }
    else {
      const path = router.resolve(`/${ts.currentLocale}` + to.path);
      if(path.name !== 'PageNotFound') {
        next({path: path.fullPath, replace: true});
        isNextCalled = true;
      }
      else {
        isLocaleInvalid = true;
      }
    }
  }

  if(!isNextCalled) {
    if(!to.name || isLocaleInvalid) {
      next({path: from.fullPath, replace: true});
    }
    else {
      next();
    }
  }
})

declare module '@vue/runtime-core' {
  interface ComponentCustomProperties {
      $router: Router,
      $route: RouteLocationNormalizedLoaded
  }
}

export default (tarnslationService: TranslationService): Router => {
  ts = tarnslationService
  return router
}
