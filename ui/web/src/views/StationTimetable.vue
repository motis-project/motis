<template>
  <div class="station-events">
    <div class="header">
      <div id="" class="back">
        <i class="icon" @click="$router.go(-1)">arrow_back</i>
      </div>
      <div class="station">{{ station.name }}</div>
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
        <div
          :class="[
            'extend-search-interval search-before',
            isUpperEnd ? 'disabled' : '',
          ]"
        >
          <a @click="changeTimeGap('EARLIER')" v-show="!isUpperEnd">Früher</a>
        </div>
        <div class="event-list">
          <div class="date-header divider">
            <span>{{
              getTimeString(filteredEvents[0].event.time, "date")
            }}</span>
          </div>
          <div v-for="timetable in filteredEvents" :key="timetable.trips[0].id">
            <div class="station-event" v-if="timetable">
              <div class="event-time">
                {{ getTimeString(timetable.event.time, "time") }}
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
              <div
                class="event-direction"
                :title="timetable.trips[0].transport.direction"
              >
                <i class="icon">arrow_forward</i
                >{{ timetable.trips[0].transport.direction }}
              </div>
              <div class="event-track"></div>
            </div>
            <div class="" v-else></div>
          </div>
          <div class="divider footer"></div>
          <div
            :class="[
              'extend-search-interval search-after',
              isBottomEnd ? 'disabled' : '',
            ]"
          >
            <a @click="changeTimeGap('LATER')" v-show="!isBottomEnd">Später</a>
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
      // departures: [] as (Event | Divider)[],
      departures: [] as Event [],
      station: {} as StationGuess,
      // filteredEvents: [] as (Event | Divider)[],
      filteredEvents: [] as Event[],
      date: 0 as number,
      direction: "BOTH",
      isDeparture: true,
      isUpperEnd: false,
      isBottomEnd: false,
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
      this.date = new Date(2020, 9, 19, new Date().getHours(), new Date().getMinutes()).valueOf() / 1000;
    } else if(this.tripIdGuess){ 
      this.date = this.tripIdGuess.time;
    }
    this.getInfo(this.stationGuess, this.isDeparture);
  },
  watch: {
    stationGuess(newValue: StationGuess) {
      this.departures = [];
      this.isUpperEnd = false;
      this.isBottomEnd = false;
      this.getInfo(newValue, this.isDeparture);
    },
  },
  methods: {
    getTimeString(timeInSeconds: number, timeOrDate: string) {
      let date = new Date(timeInSeconds * 1000);
      if (timeOrDate === "time") {
        return (
          ("0" + date.getHours()).slice(-2) +
          ":" +
          ("0" + date.getMinutes()).slice(-2)
        );
      } else if (timeOrDate === "date") {
        return (
          date.getDate() +
          "." +
          ("0" + (date.getMonth() + 1)).slice(-2) +
          "." +
          date.getFullYear()
        );
      }
    },
    getDepartures(isDeparture: boolean) {
      if (isDeparture && this.departures) {
        this.filteredEvents = this.departures.filter(
          (event) => event.type == "DEP"
        );
      } else {
        this.filteredEvents = this.departures.filter(
          (event) => event.type == "ARR"
        );
      }
      this.isDeparture = isDeparture;
    },
    changeTimeGap(change: string) {
      this.direction = change;
      if (change == "EARLIER") {
        this.date = this.departures[0].event.time;
      } else if (change == "LATER") {
        this.date = this.departures[this.departures.length - 1].event.time;
      }
      this.getInfo(this.stationGuess, this.isDeparture, change === "EARLIER");
    },

    getInfo(
      newValue: StationGuess,
      isDeparture: boolean,
      clickEarlier: boolean | null = null
    ) {
      this.$postService
        .getDeparturesResponse(newValue.id, true, this.direction, 20, this.date)
        .then((data) => {
          if (clickEarlier === null) {
            this.departures = data.events;
          } else if (!clickEarlier) {
            this.departures = this.departures.concat(data.events);
          } else if (clickEarlier) {
            this.departures = data.events.concat(this.departures);
          }
          if (data.events.length === 0) {
            if (clickEarlier === false) {
              this.isBottomEnd = true;
            } else if (clickEarlier === true) {
              this.isUpperEnd = true;
            }
          }
          this.station = data.station;
          this.getDepartures(isDeparture);
        });
    },
  },
});

interface Divider {}
</script>