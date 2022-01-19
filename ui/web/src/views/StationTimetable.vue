<template>
  <LoadingBar v-if="!isContentLoaded"></LoadingBar>
  <div class="station-events" v-else>
    <div class="header">
      <div id="" class="back">
        <i class="icon" @click="$router.go(-1)">arrow_back</i>
      </div>
      <div class="station">
        {{ station.name }}
      </div>
      <div class="event-type-picker">
        <div>
          <input
            type="radio"
            id="station-departures"
            name="station-event-types"
            checked />
          <label
            for="station-departures"
            @click="getDepartures(true)">{{ $t.departure }}</label>
        </div>
        <div>
          <input
            type="radio"
            id="station-arrivals"
            name="station-event-types" />
          <label
            for="station-arrivals"
            @click="getDepartures(false)">{{ $t.arrival }}</label>
        </div>
      </div>
    </div>
    <div class="events">
      <div class="">
        <div
          :class="[
            'extend-search-interval search-before',
            isUpperEnd ? 'disabled' : '',
          ]">
          <a @click="changeTimeGap('EARLIER')" v-show="!isUpperEnd"> {{ $t.earlier }} </a>
        </div>
        <div class="event-list">
          <div class="date-header divider">
            <span>{{
              getDateString(filteredEvents[0].event.time)
            }}</span>
          </div>
          <div v-for="(timetable, index) in filteredEvents" :key="timetable.trips[0].id">
            <div class="station-event" v-if="timetable">
              <div class="event-time">
                {{ getTimeString(timetable.event.time) }}
              </div>
              <div class="event-train">
                <span>
                  <TransportTypeBox
                    :transport="timetable.trips[0].transport"
                    :trip="timetable.trips[0].id">
                  </TransportTypeBox>
                </span>
              </div>
              <div
                class="event-direction"
                :title="timetable.trips[0].transport.direction">
                <i class="icon">arrow_forward</i>{{ timetable.trips[0].transport.direction }}
              </div>
              <div class="event-track"></div>
            </div>
            <div class="date-header divider" v-if="separators.includes(index + 1)">
              <span>{{ getDateString(filteredEvents[index + 1].event.time) }}</span>
            </div>
            <div class="" v-else></div>
          </div>
          <div class="divider footer"></div>
          <div
            :class="[
              'extend-search-interval search-after',
              isBottomEnd ? 'disabled' : '',
            ]">
            <a @click="changeTimeGap('LATER')" v-show="!isBottomEnd"> {{ $t.later }} </a>
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
import LoadingBar from "../components/LoadingBar.vue"

export default defineComponent({
  name: "StationTimetable",
  components: {
    TransportTypeBox,
    LoadingBar
  },
  props: {
    stationGuess: {
      type: Object as PropType<StationGuess>,
      required: true
    },
    tripIdGuess: {
      type: Object as PropType<TripInfoId>,
      required: false,
    }
  },
  data() {
    return {
      departures: [] as Event [],
      station: {} as StationGuess,
      filteredEvents: [] as Event[],
      date: 0 as number,
      direction: "BOTH",
      isDeparture: true,
      isUpperEnd: false,
      isBottomEnd: false,
      isContentLoaded: false,
      separators: [] as number []
    };
  },
  watch: {
    stationGuess(newValue: StationGuess) {
      this.departures = [];
      this.isUpperEnd = false;
      this.isBottomEnd = false;
      this.getInfo(newValue, this.isDeparture);
    },
  },
  created() {
    if(!this.tripIdGuess){
      this.date = this.$ds.dateTimeInSeconds;
    }
    else if(this.tripIdGuess){
      this.date = this.tripIdGuess.time;
    }
    this.getInfo(this.stationGuess, this.isDeparture);
  },
  methods: {
    getTimeString(timeInSeconds: number) {
      return this.$ds.getTimeString(timeInSeconds * 1000);
    },
    getDateString(timeInSeconds: number) {
      return this.$ds.getDateString(timeInSeconds * 1000);
    },
    getDepartures(isDeparture: boolean) {
      if (isDeparture && this.departures) {
        this.filteredEvents = this.departures.filter(
          (event) => event.type === "DEP"
        );
      }
      else {
        this.filteredEvents = this.departures.filter(
          (event) => event.type === "ARR"
        );
      }
      this.getSeparator(this.filteredEvents);
      this.isDeparture = isDeparture;
    },
    changeTimeGap(change: string) {
      this.direction = change;
      if (change === "EARLIER") {
        this.date = this.departures[0].event.time;
      }
      else if (change === "LATER") {
        this.date = this.departures[this.departures.length - 1].event.time;
      }
      this.getInfo(this.stationGuess, this.isDeparture, change === "EARLIER");
    },
    getSeparator(events : Event []) {
      this.separators = [];
      for (let i = 1; i < events.length; i++) {
        let earlier = new Date(events[i - 1].event.time * 1000);
        let later = new Date(events[i].event.time * 1000);
        if (earlier.getDate() < later.getDate() || earlier.getMonth() < later.getMonth() || earlier.getFullYear() < later.getFullYear()) {
          this.separators.push(i);
        }
      }
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
          }
          else if (!clickEarlier) {
            this.departures = this.departures.concat(data.events);
          }
          else if (clickEarlier) {
            this.departures = data.events.concat(this.departures);
          }
          if (data.events.length === 0) {
            if (clickEarlier === false) {
              this.isBottomEnd = true;
            }
            else if (clickEarlier === true) {
              this.isUpperEnd = true;
            }
          }
          this.station = data.station;
          this.getDepartures(isDeparture);
          this.isContentLoaded = true;
        }).catch(() => this.$router.back());
    },
  },
});
</script>
