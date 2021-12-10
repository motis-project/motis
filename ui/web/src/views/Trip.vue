<template>
  <LoadingBar v-if="!isContentLoaded"></LoadingBar>
  <div v-else class="connection-details trip-view">
    <div class="connection-info">
      <div class="header">
        <div class="back" @click="$router.back()"><i class="icon">arrow_back</i></div>
        <div class="details">
          <div class="date">{{ date }}</div>
          <div class="connection-times">
            <div class="times">
              <div class="connection-departure">{{ departure }}</div>
              <div class="connection-arrival">{{ arrival }}</div>
            </div>
            <div class="locations">
              <div>{{ firstStopName }}</div>
              <div>{{ lastStopName }}</div>
            </div>
          </div>
          <div class="summary">
            <span class="duration"><i class="icon">schedule</i>{{ duration }}</span
            ><span class="interchanges"><i class="icon">transfer_within_a_station</i>{{ changes }}</span>
          </div>
        </div>
        <div class="actions"></div>
      </div>
    </div>
    <div class="connection-journey" id="sub-connection-journey">
      <div
        v-for="(transport, tIndex) in transports"
        :key="transport.line_id"
        :class="`train-detail train-class-${transport.clasz}`"
      >
        <div class="top-border"></div>
        <TransportTypeBox :transport="transport"></TransportTypeBox>
        <div class="first-stop">
          <div :class="['stop', getPastOrFuture(getFirstStop(transport).departure.time)]">
            <div class="timeline train-color-border"></div>
            <div class="time">
              <span :class="getPastOrFuture(getFirstStop(transport).departure.time)">{{ departure }}</span>
            </div>
            <div class="delay"></div>
            <div class="station">
              <span>{{ firstStopName }}</span>
            </div>
          </div>
        </div>
        <div :class="['direction', getPastOrFuture(getFirstStop(transport).departure.time)]">
          <div class="timeline train-color-border"></div>
          <i class="icon">arrow_forward</i>{{ lastStopName }}
        </div>
        <div :class="['intermediate-stops-toggle', 'clickable', getPastOrFuture(getFirstStop(transport).departure.time)]" @click="areStopsExpanded[tIndex] = !areStopsExpanded[tIndex]">
          <div class="timeline-container">
            <div class="timeline train-color-border bg"></div>
            <div class="timeline train-color-border progress" :style="`height: ${getStopProgress(getFirstStop(transport))}%`"></div>
          </div>
          <div class="expand-icon">
            <i class="icon">{{ areStopsExpanded[tIndex] ? "expand_more" : "expand_less" }}</i
            ><i class="icon">{{ areStopsExpanded[tIndex] ? "expand_less" : "expand_more" }}</i>
          </div>
          <span>Fahrt {{ content.stops.length - 2 }} Stationen ({{ duration }})</span>
        </div>
        <div :class="['intermediate-stops', areStopsExpanded[tIndex] ? 'expanded' : '']" v-show="areStopsExpanded[tIndex]">
          <div v-for="stop in getIntermediateStops(transport)" :key="stop.station" :class="['stop', getPastOrFuture(stop.departure.time)]">
            <div class="timeline train-color-border bg"></div>
            <div class="timeline train-color-border progress" :style="`height: ${getStopProgress(stop)}%`"></div>
            <div class="time">
              <div class="arrival">
                <span :class="getPastOrFuture(stop.arrival.time)">{{ getTimeString(stop.arrival.time) }}</span>
              </div>
              <div class="departure">
                <span :class="getPastOrFuture(stop.departure.time)">{{ getTimeString(stop.departure.time) }}</span>
              </div>
            </div>
            <div class="delay">
              <div class="arrival"></div>
              <div class="departure"></div>
            </div>
            <div class="station">
              <span>{{ stop.station.name }}</span>
            </div>
          </div>
        </div>
        <div class="last-stop">
          <div :class="['stop', getPastOrFuture(getLastStop(transport).arrival.time)]">
            <div class="timeline train-color-border"></div>
            <div class="time">
              <span :class="getPastOrFuture(getLastStop(transport).arrival.time)">{{ arrival }}</span>
            </div>
            <div class="delay"></div>
            <div class="station">
              <span>{{ lastStopName }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, PropType } from "vue";
import Trip from "../models/Trip";
import TripResponseContent from "../models/TripResponseContent";
import TransportTypeBox from "../components/TransportTypeBox.vue";
import Transport from "../models/Transport";
import Stop from "../models/Stop";
import LoadingBar from "../components/LoadingBar.vue"

export default defineComponent({
  components: {
    TransportTypeBox,
    LoadingBar
  },
  props: {
    trip: {
      type: Object as PropType<Trip>,
      required: true,
    },
    initContent: {
      type: Object as PropType<TripResponseContent>,
      required: false
    }
  },
  data() {
    return {
      content: {} as TripResponseContent,
      isContentLoaded: false,
      areStopsExpanded: [] as Boolean[],
    };
  },
  computed: {
    date: function (): string {
      return new Date(this.content.trips[0].id.time * 1000).toLocaleString("de-DE", {
        month: "2-digit",
        year: "numeric",
        day: "numeric",
      });
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
        res += `${time.getDate() - 1}d`;
      }
      if (res != "") {
        res += " ";
      }
      if (time.getHours() > 1) {
        res += `${time.getHours() - 1}h`;
      }
      if (res != "") {
        res += " ";
      }
      if (time.getMinutes() > 0) {
        res += `${time.getMinutes()}min`;
      }
      return res;
    },
    changes: function (): string {
      let res = "";
      if (this.content.trips.length == 1) {
        res = "Keine Umstiege";
      } else if (this.content.trips.length == 2) {
        res = "1 Umstieg";
      } else if (this.content.trips.length > 2) {
        res = `${this.content.trips.length} Umstiege`;
      }
      return res;
    },
    transports: function (): Transport[] {
      return this.content.transports.map((t) => t.move);
    },
  },
  created() {
    if(this.initContent) {
      this.content = this.initContent
    }
    this.$postService.getTripResponce(this.trip).then((data) => {
      if (data.trips.length > 0 && data.trips[0].id.station_id === this.trip.station_id) {
        this.content = data;

      }
    });
  },
  methods: {
    getFirstStop(transport: Transport) {
      return this.content.stops[0];
    },
    getLastStop(transport: Transport) {
      return this.content.stops[this.content.stops.length - 1];
    },
    getIntermediateStops(transport: Transport) {
      return this.content.stops.slice(transport.range.from + 1, transport.range.to - 1);
    },
    getTimeString(timeInSeconds: number) {
      let date = new Date(timeInSeconds * 1000);
      return date.getHours() + ":" + ("0" + date.getMinutes()).slice(-2);
    },
    getPastOrFuture(timeInSeconds: number) {
      let date = new Date(timeInSeconds * 1000);
      return date < new Date() ? 'past' : 'future';
    },
    getStopProgress(stop: Stop) {
      let index = this.content.stops.indexOf(stop);
      let nextStop = this.content.stops[index + 1];
      let diff = nextStop.arrival.time - stop.departure.time;
      let diffWithCurrent = nextStop.arrival.time - (new Date().valueOf() / 1000);
      if(diffWithCurrent < 0) {
        return 100;
      }
      else if(diffWithCurrent > diff) {
        return 0;
      }
      else {
        return (diffWithCurrent / diff) * 100;
      }
    }
  },
  watch: {
    content: function(newValue: TripResponseContent) {
        let expand = this.$route.name === "Trip";
        this.content.transports.forEach((t) => this.areStopsExpanded.push(expand));
        this.isContentLoaded = true;
    }
  }
});
</script>
