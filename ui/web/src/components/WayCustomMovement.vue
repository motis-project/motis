<template>
  <div class="train-detail train-class-walk trailing-walk">
    <div class="top-border"></div>
    <div>
      <div class="train-box train-class-walk">
        <svg class="train-icon">
          <use :xlink:href="`#${customMovement.mumo_type}`"></use>
        </svg>
      </div>
    </div>
    <div class="first-stop">
      <div :class="['stop', getPastOrFuture($ds.simulationDate, stops[0].departure.time)]">
        <div class="timeline train-color-border"></div>
        <div class="time">
          <span :class="getPastOrFuture($ds.simulationDate, stops[0].departure.time)">{{ $ds.getTimeString(stops[0].departure.time * 1000) }}</span>
        </div>
        <div class="delay"></div>
        <div class="station">
          <span>{{ startStationName }}</span>
        </div>
      </div>
    </div>
    <div class="intermediate-stops-toggle">
      <div class="timeline-container">
        <div class="timeline train-color-border bg"></div>
        <div
          class="timeline train-color-border progress"
          :style="`height: ${getStopProgress(stops[0])}%`"></div>
      </div>
      <div class="expand-icon"></div>
      <span>{{ (customMovement.mumo_type === "foot" ? $t.walk : $t[customMovement.mumo_type]) + " (" + getReadableDuration(stops[0].departure.time, stops[1].arrival.time, $ts) + ")" }}</span>
    </div>
    <div class="last-stop">
      <div :class="['stop', getPastOrFuture($ds.simulationDate, stops[1].arrival.time)]">
        <div class="timeline train-color-border"></div>
        <div class="time">
          <span :class="getPastOrFuture($ds.simulationDate, stops[1].arrival.time)">{{ $ds.getTimeString(stops[1].arrival.time * 1000) }}</span>
        </div>
        <div class="delay"></div>
        <div class="station">
          <span class="virtual">{{ endStationName }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, PropType } from "vue";
import CustomMovement from "../models/CustomMovement";
import WayMixin from "../mixins/WayMixin";
import Stop from "../models/Stop";

export default defineComponent({
  name: "WayCustomMovement",
  mixins: [ WayMixin ],
  props: {
    customMovement: {
      type: Object as PropType<CustomMovement>,
      required: true,
    },
    stops: {
      type: Object as PropType<Stop[]>,
      required: true
    },
    startStationName: {
      type: String as PropType<string>,
      required: true
    },
    endStationName: {
      type: String as PropType<string>,
      required: true
    }
  },
  methods: {
    getStopProgress(stop: Stop) {
      return this.getProgress(stop, this.stops[this.stops.indexOf(stop) + 1], this.$ds.simulationTimeInSeconds);
    }
  }
});
</script>
