<template>
  <div :class="`train-detail train-class-${transport.clasz}`">
    <div class="top-border"></div>
    <TransportTypeBox :transport="transport" :trip="trip"></TransportTypeBox>
    <div class="train-top-line" v-if="additionalMove">
      <span>{{ additionalMove + " " + $t.walk }}</span>
    </div>
    <div class="first-stop">
      <div :class="['stop', getPastOrFuture($ds.simulationDate, stops[0].departure.time)]">
        <div class="timeline train-color-border"></div>
        <div class="time">
          <span :class="getPastOrFuture($ds.simulationDate, stops[0].departure.time)">{{ $ds.getTimeString(stops[0].departure.time * 1000) }}</span>
        </div>
        <div class="delay"></div>
        <div class="station">
          <span @click="goToStop($router, stops[0])">{{ stops[0].station.name }}</span>
        </div>
      </div>
    </div>
    <div :class="['direction', getPastOrFuture($ds.simulationDate, stops[0].departure.time)]">
      <div class="timeline train-color-border"></div>
      <i class="icon">arrow_forward</i>{{ transport.direction }}
    </div>
    <div :class="['intermediate-stops-toggle', 'clickable', getPastOrFuture($ds.simulationDate, stops[0].departure.time)]" @click="areStopsExpanded = !areStopsExpanded">
      <div class="timeline-container">
        <div class="timeline train-color-border bg"></div>
        <div class="timeline train-color-border progress" :style="`height: ${getStopProgress(stops[0])}%`"></div>
      </div>
      <div class="expand-icon" :style="'visibility:' + (stops.length > 2 ? 'visible' : 'hidden')">
        <i class="icon">{{ areStopsExpanded ? "expand_more" : "expand_less" }}</i><i class="icon">{{ areStopsExpanded ? "expand_less" : "expand_more" }}</i>
      </div>
      <span>{{ $ts.countTranslate("stop", stops.length - 2) }} ({{ getReadableDuration(stops[0].departure.time, stops[stops.length - 1].arrival.time, $ts) }})</span>
    </div>
    <div :class="['intermediate-stops', areStopsExpanded ? 'expanded' : '']" v-show="areStopsExpanded">
      <div v-for="stop in getIntermediateStops(transport)" :key="stop.station" :class="['stop', getPastOrFuture($ds.simulationDate, stop.departure.time)]">
        <div class="timeline train-color-border bg"></div>
        <div class="timeline train-color-border progress" :style="`height: ${getStopProgress(stop)}%`"></div>
        <div class="time">
          <div class="arrival">
            <span :class="getPastOrFuture($ds.simulationDate, stop.arrival.time)">{{ $ds.getTimeString(stop.arrival.time * 1000) }}</span>
          </div>
          <div class="departure">
            <span :class="getPastOrFuture($ds.simulationDate, stop.departure.time)">{{ $ds.getTimeString(stop.departure.time * 1000) }}</span>
          </div>
        </div>
        <div class="delay">
          <div class="arrival"></div>
          <div class="departure"></div>
        </div>
        <div class="station">
          <span @click="goToStop($router, stop)">{{ stop.station.name }}</span>
        </div>
      </div>
    </div>
    <div class="last-stop">
      <div :class="['stop', getPastOrFuture($ds.simulationDate, stops[stops.length - 1].arrival.time)]">
        <div class="timeline train-color-border"></div>
        <div class="time">
          <span :class="getPastOrFuture($ds.simulationDate, stops[stops.length - 1].arrival.time)">{{ $ds.getTimeString(stops[stops.length - 1].arrival.time * 1000) }}</span>
        </div>
        <div class="delay"></div>
        <div class="station">
          <span @click="goToStop($router, stops[stops.length - 1])">{{ stops[stops.length - 1].station.name }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, PropType } from 'vue'
import TransportTypeBox from './TransportTypeBox.vue'
import WayMixin from "../mixins/WayMixin"
import Transport from "../models/Transport"
import Stop from '../models/Stop'
import Trip from '../models/Trip'

export default defineComponent({
  name: "WayTransport",
  components: {
    TransportTypeBox
  },
  mixins: [WayMixin],
  props: {
    transport: {
      type: Object as PropType<Transport>,
      required: true
    },
    stops: {
      type: Object as PropType<Stop[]>,
      required: true
    },
    areStopsInitialExpanded: {
      type: Boolean as PropType<boolean>,
      required: true
    },
    additionalMove: {
      type: String as PropType<string>,
      required: false
    },
    trip: {
      type: Object as PropType<Trip>,
      required: false
    }
  },
  data() {
    return {
      areStopsExpanded: this.areStopsInitialExpanded
    }
  },
  methods: {
    getIntermediateStops() {
      return this.stops.slice(1, this.stops.length - 1);
    },
    getStopProgress(stop: Stop) {
      return this.getProgress(stop, this.stops[this.stops.indexOf(stop) + 1], this.$ds.simulationTimeInSeconds);
    }
  }
})
</script>
