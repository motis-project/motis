<template>
  <LoadingBar v-if="!isContentLoaded"></LoadingBar>
  <div v-else class="connection-details trip-view">
    <div class="connection-info">
      <div class="header">
        <div class="back" @click="$router.back()">
          <i class="icon">arrow_back</i>
        </div>
        <div class="details">
          <div class="date">
            {{ date }}
          </div>
          <div class="connection-times">
            <div class="times">
              <div class="connection-departure">
                {{ departure }}
              </div>
              <div class="connection-arrival">
                {{ arrival }}
              </div>
            </div>
            <div class="locations">
              <div>{{ firstStopName }}</div>
              <div>{{ lastStopName }}</div>
            </div>
          </div>
          <div class="summary">
            <span class="duration"><i class="icon">schedule</i>{{ duration }}</span><span class="interchanges"><i class="icon">transfer_within_a_station</i>{{ changes }}</span>
          </div>
        </div>
        <div class="actions"></div>
      </div>
    </div>
    <div class="connection-journey" id="sub-connection-journey">
      <div
        v-for="(transport) in transports"
        :key="transport.line_id">
        <WayTransport
          v-if="transport.move_type === 'Transport'"
          :areStopsInitialExpanded="$route.name === 'Trip'"
          :transport="transport.move"
          :stops="getStopsForTransport(transport.move)"></WayTransport>
        <WayCustomMovement
          v-else
          :customMovement="transport.move"
          :stops="getStopsForTransport(transport.move)"
          :startStationName="determineStartStationNameForCustomMovement(transport.move)"
          :endStationName="determineEndStationNameForCustomMovement(transport.move)"></WayCustomMovement>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, PropType } from "vue";
import Trip from "../models/Trip";
import TripResponseContent, { Move } from "../models/TripResponseContent";
import Transport from "../models/Transport";
import LoadingBar from "../components/LoadingBar.vue"
import WayTransport from "../components/WayTransport.vue"
import WayCustomMovement from "../components/WayCustomMovement.vue"
import CustomMovement from "../models/CustomMovement";

export default defineComponent({
  name: "Trip",
  components: {
    LoadingBar,
    WayTransport,
    WayCustomMovement
  },
  props: {
    trip: {
      type: Object as PropType<Trip>,
      required: false,
    },
    initContent: {
      type: Object as PropType<TripResponseContent>,
      required: false,
    },
    startStationName: {
      type: String as PropType<string>,
      required: false
    },
    endStationName: {
      type: String as PropType<string>,
      required: false
    }
  },
  data() {
    return {
      content: {} as TripResponseContent,
      isContentLoaded: false,
    };
  },
  computed: {
    date: function (): string {
      return this.$ds.getDateString(this.content.trips[0].id.time * 1000);
    },
    departure: function (): string {
      return this.getTimeString(this.content.stops[0].departure.time);
    },
    arrival: function (): string {
      return this.getTimeString(this.content.stops[this.content.stops.length - 1].arrival.time);
    },
    firstStopName: function (): string {
      return this.content.stops[0].station.name;
    },
    lastStopName: function (): string {
      return this.content.stops[this.content.stops.length - 1].station.name;
    },
    duration: function (): string {
      let time = new Date(
        (this.content.stops[this.content.stops.length - 1].arrival.time - this.content.stops[0].departure.time) * 1000
      );
      let res = "";
      if (time.getDate() > 1) {
        res += this.$ts.formatTranslate("days", (time.getDate() - 1).toString());
      }
      if (res !== "") {
        res += " ";
      }
      if (time.getHours() > 1) {
        res += this.$ts.formatTranslate("hours", (time.getHours() - 1).toString());
      }
      if (res !== "") {
        res += " ";
      }
      if (time.getMinutes() > 0) {
        res += this.$ts.formatTranslate("minutes", time.getMinutes().toString());
      }
      return res;
    },
    changes: function (): string {
      return this.$ts.countTranslate("changes", this.content.trips.length - 1);
    },
    transports: function (): Move[] {
      return this.content.transports;
    },
  },
  watch: {
    content: function() {
      this.isContentLoaded = true;
    },
    trip: function() {
      this.sendRequest();
    }
  },
  created() {
    if(this.initContent) {
      this.content = this.initContent
    }
    else {
      this.sendRequest();
    }
  },
  methods: {
    getTimeString(timeInSeconds: number) {
      return this.$ds.getTimeString(timeInSeconds * 1000);
    },
    sendRequest() {
      if(this.trip !== undefined) {
        this.$postService.getTripResponce(this.trip).then((data) => {
          if (this.trip !== undefined && data.trips.length > 0 && data.trips[0].id.station_id === this.trip.station_id) {
            this.content = data;
          }
        }).catch(() => this.$router.back());
      }
    },
    getStopsForTransport(transport: Transport | CustomMovement) {
      return this.content.stops.slice(transport.range.from, transport.range.to + 1);
    },
    determineStartStationNameForCustomMovement(customMovement: CustomMovement) {
      if(this.transports.map(t => t.move).indexOf(customMovement) === 0 && this.startStationName !== undefined) {
        return this.startStationName;
      }
      else if(customMovement.mumo_type === "car") {
        return "Parkplatz";
      }
      else {
        return this.getStopsForTransport(customMovement)[0].station.name;
      }
    },
    determineEndStationNameForCustomMovement(customMovement: CustomMovement) {
      if(this.transports.map(t => t.move).indexOf(customMovement) === this.transports.length - 1 && this.endStationName !== undefined) {
        return this.endStationName;
      }
      else if(customMovement.mumo_type === "car") {
        return "Parkplatz";
      }
      else {
        return this.getStopsForTransport(customMovement)[1].station.name;
      }
    }
  }
});
</script>
