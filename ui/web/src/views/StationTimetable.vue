<template>
  <div class="station-events">
    <div class="header">
      <div id="" class="back">
        <i class="icon" @click="$router.go(-1)">arrow_back</i>
      </div>
      <div class="station">{{ stationGuess.name }}</div>
      <div class="event-type-picker">
        <div>
          <input
            type="radio"
            id="station-departures"
            name="station-event-types"
            checked
          />
          <label for="station-departures" @click="getDepartures(true)"
            >{{$t.departure}}</label
          >
        </div>
        <div>
          <input
            type="radio"
            id="station-arrivals"
            name="station-event-types"
          />
          <label for="station-arrivals" @click="getDepartures(false)"
            >{{$t.arrival}}</label
          >
        </div>
      </div>
    </div>
    <div class="events">
      <div class="">
        <div class="extend-search-interval search-before disabled"></div>
        <div class="event-list">
          <div
            class="station-event"
            v-for="timetable in filteredEvents"
            :key="timetable.trips[0].id"
          >
            <div class="event-time">
              {{ getTimeString(timetable.event.time) }}
            </div>
            <div class="event-train">
              <span>
                <TransportTypeBox
                  :transport="timetable.trips[0].transport"
                  :trip="timetable.trips[0].id"
                >
                </TransportTypeBox>
              </span>
            </div>
            <div class="event-direction" :title="timetable.trips[0].transport.direction">
              <i class="icon">arrow_forward</i>{{ timetable.trips[0].transport.direction }}
            </div>
            <div class="event-track"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import StationGuess from "../models/StationGuess";
import { defineComponent, PropType } from "vue";
import Event from "../models/DepartureTimetable";
import TransportTypeBox from "../components/TransportTypeBox.vue";
import { TripInfoId } from "../models/TrainGuess";

export default defineComponent({
  name: "StationTimetable",
  data() {
    return {
      departures: [] as Event[],
      station: {} as StationGuess,
      filteredEvents: [] as Event[],
    };
  },
  components: {
    TransportTypeBox,
  },
  props: {
    stationGuess: {
      type: Object as PropType<StationGuess>,
      required: true
    },
    tripIdGuess: {
      type: Object as PropType<TripInfoId>,
      required: false
    }
  },
  created() {
    if(!this.tripIdGuess){
      this.$postService
        .getDeparturesResponse(
          this.stationGuess.id,
          true,
          "BOTH",
          20,
          new Date(2020, 9, 19, 18, 0).valueOf() / 1000
        )
        .then((data) => {
          this.departures = data.events;
          this.station = data.station;
          this.getDepartures(true);
        });
    } else if(this.tripIdGuess){ 
      this.$postService
        .getDeparturesResponse(
          this.stationGuess.id,
          true,
          "BOTH",
          20,
          this.tripIdGuess.time
        )
        .then((data) => {
          this.departures = data.events;
          this.station = data.station;
          this.getDepartures(true);
        });
     }
  },
  methods: {
    getTimeString(timeInSeconds: number) {
      let date = new Date(timeInSeconds * 1000);
      return date.getHours() + ":" + ("0" + date.getMinutes()).slice(-2);
    },
    getDepartures(isDeparture: Boolean) {
      if (isDeparture) {
        this.filteredEvents = this.departures.filter(
          (event) => event.type == "DEP"
        );
      } else {
        this.filteredEvents = this.departures.filter(
          (event) => event.type == "ARR"
        );
      }
    },
  },
});
</script>