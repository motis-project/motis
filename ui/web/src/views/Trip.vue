<template>
  <LoadingBar v-if="!isContentLoaded"></LoadingBar>
  <div v-else :class="['connection-details', isTripView ? 'trip-view' : '']">
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
    <div class="connection-journey" :id="isTripView ? 'sub-connection-journey' : 'connection-journey'">
      <template
        v-for="(transport, tIndex) in transports"
        :key="transport.line_id">
        <WayTransport
          v-if="transport.move_type === 'Transport'"
          :areStopsInitialExpanded="isTripView"
          :transport="transport.move"
          :stops="getStopsForTransport(transport.move)"
          :additionalMove="additionalMoves[tIndex]"
          :trip="getTripForTransport(transport.move)"></WayTransport>
        <template v-else-if="transport.move.mumo_id === -1"></template>
        <WayCustomMovement
          v-else
          :customMovement="transport.move"
          :stops="getStopsForTransport(transport.move)"
          :startStationName="determineStartStationNameForCustomMovement(transport.move)"
          :endStationName="determineEndStationNameForCustomMovement(transport.move)"></WayCustomMovement>
      </template>
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
import WayMixin from "../mixins/WayMixin"

export default defineComponent({
  name: "Trip",
  components: {
    LoadingBar,
    WayTransport,
    WayCustomMovement
  },
  mixins: [ WayMixin ],
  props: {
    trip: {
      type: Object as PropType<Trip>,
      required: false,
    },
    index: {
      type: Number as PropType<number>,
      required: false
    }
  },
  data() {
    return {
      content: {} as TripResponseContent,
      isContentLoaded: false,
      startStationName: undefined as (string | undefined),
      endStationName: undefined as (string | undefined),
      additionalMoves: [] as (string | undefined)[]
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
      return this.index !== undefined ? this.startStationName as string : this.content.stops[0].station.name;
    },
    lastStopName: function (): string {
      return this.index !== undefined ? this.endStationName as string : this.content.stops[this.content.stops.length - 1].station.name;
    },
    duration: function (): string {
      return this.getReadableDuration(this.content.stops[0].departure.time, this.content.stops[this.content.stops.length - 1].arrival.time, this.$ts);
    },
    changes: function (): string {
      return this.$ts.countTranslate("changes", this.content.trips.length - 1);
    },
    transports: function (): Move[] {
      return this.content.transports;
    },
    isTripView() {
      return this.$route.name === "Trip";
    }
  },
  watch: {
    content: function() {
      this.isContentLoaded = true;
      let additionalMoveForNext: string | undefined = undefined;
      for(let t of this.content.transports) {
        this.additionalMoves.push(additionalMoveForNext)
        additionalMoveForNext = undefined;
        if("mumo_id" in t.move && t.move.mumo_id === -1) {
          additionalMoveForNext =  this.getReadableDuration(this.content.stops[t.move.range.from].departure.time, this.content.stops[t.move.range.to].arrival.time, this.$ts);
        }
      }
    },
    trip() {
      this.getData();
    },
    index() {
      this.getData();
    }
  },
  created() {
    this.getData();
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
      let transportIndex = this.content.transports.map(t => t.move).indexOf(transport);
      let nextTransport = transportIndex === this.content.transports.length - 1 ? transport : this.content.transports[transportIndex + 1].move;
      let addOne = (!("mumo_id" in nextTransport) || nextTransport.mumo_id !== -1) && (!("mumo_id" in transport) || transport.mumo_id !== -1);
      return this.content.stops.slice(transport.range.from, transport.range.to + (addOne ? 1 : 0));
    },
    determineStartStationNameForCustomMovement(customMovement: CustomMovement) {
      let start = this.getStopsForTransport(customMovement)[0].station.name;
      let ts = this.transports.map(t => t.move);
      if(ts.indexOf(customMovement) === 0 && this.startStationName !== undefined) {
        start = this.startStationName
      }
      else if(ts.indexOf(customMovement) === 1 && "mumo_type" in ts[0] && ts[0].mumo_type === "car"
        || ts.indexOf(customMovement) === ts.length - 1 && "mumo_type" in customMovement && customMovement.mumo_type === "car") {
        start = this.$t.parking
      }
      return start;
    },
    determineEndStationNameForCustomMovement(customMovement: CustomMovement) {
      let stops = this.getStopsForTransport(customMovement);
      let destination = stops[stops.length - 1].station.name;
      let ts = this.transports.map(t => t.move);
      let lastT = ts[ts.length - 1];

      if(ts.indexOf(customMovement) === ts.length - 1 && this.endStationName !== undefined) {
        destination = this.endStationName;
      }
      else if(ts.indexOf(customMovement) === ts.length - 2 && "mumo_type" in lastT && lastT.mumo_type === "car"
        || ts.indexOf(customMovement) === 0 && "mumo_type" in customMovement && customMovement.mumo_type === "car") {
        destination = this.$t.parking
      }
      return destination;
    },
    getData() {
      if(this.index !== undefined && this.$store.state.connections.length > 0
        && this.index < this.$store.state.connections.length && this.index >= 0) {
        this.content = this.$store.state.connections[this.index];
        if(!this.$store.state.areConnectionsDropped) {
          this.startStationName = this.$store.state.startInput.name;
          this.endStationName = this.$store.state.destinationInput.name;
        }
      }
      else if(this.trip !== undefined) {
        this.sendRequest();
      }
      else {
        this.$router.push({name: "ConnectionSearch"});
      }
    },
    getTripForTransport(transport: Transport) {
      if(!this.isTripView) {
        return this.content.trips.filter(t => t.id.line_id === transport.line_id)[0].id;
      }
      return undefined;
    }
  }
});
</script>
