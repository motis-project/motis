<template>
  <div class="trip-search">
    <div class="header">
      <div id="trip-search-form">
        <div class="pure-g gutters">
          <InputField
            class="pure-u-1 pure-u-sm-1-2 train-nr"
            :labelName="$t.trainNr"
            iconType="train"
            :showLabel="true"
            :showAutocomplete="false"
            @inputChanged="setCurrentTrainNumber"></InputField>
        </div>
        <div class="pure-g gutters">
          <Calendar class="pure-u-1 pure-u-sm-12-24 to-location" @dateChanged="setNewDate"></Calendar>
          <TimeInputField @timeChanged="setNewTime" class="pure-u-1 pure-u-sm-12-24"></TimeInputField>
        </div>
      </div>
    </div>
    <LoadingBar v-if="contentLoadingState === LoadingState.Loading"></LoadingBar>
    <div class="trips" v-else-if="contentLoadingState === LoadingState.Loaded">
      <ul style="list-style-type: none; margin-left: -40px" v-show="areGuessesDisplayed">
        <li class="trip" v-for="trip in trainGuesses" :key="trip">
          <div class="trip-train">
            <span>
              <TransportTypeBox :transport="trip.trip_info.transport" :trip="trip.trip_info.id"></TransportTypeBox>
            </span>
          </div>
          <div class="trip-time">
            <div class="time">
              {{ setTimeToDisplay(trip.trip_info.id.time) }}
            </div>
            <div class="date" v-if="checkDay(trip.trip_info.id.time)">
              {{ setDateToDisplay(trip.trip_info.id.time) }}
            </div>
          </div>
          <div class="trip-first-station">
            <div class="station" title="" @click="goToFirstStation(trip)">
              {{ trip.first_station.name }}
            </div>
            <div class="direction" :title="trip.trip_info.transport.direction">
              <i class="icon">arrow_forward</i>
              {{ trip.trip_info.transport.direction }}
            </div>
          </div>
        </li>
      </ul>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import Calendar from "../components/Calendar.vue";
import TimeInputField from "../components/TimeInputField.vue";
import InputField from "../components/InputField.vue";
import Trips from "../models/TrainGuess";
import TransportTypeBox from "../components/TransportTypeBox.vue";
import LoadingBar, {LoadingState} from "../components/LoadingBar.vue"

export default defineComponent({
  name: "TrainSearch",
  components: {
    Calendar,
    TimeInputField,
    InputField,
    TransportTypeBox,
    LoadingBar
  },
  data() {
    return {
      currentTrainInput: -1 as number,
      areGuessesDisplayed: false as boolean,
      currentDate: {} as Date,
      trainGuesses: [] as Trips[],
      contentLoadingState: LoadingState.NothingToShow,
      LoadingState: LoadingState
    };
  },
  watch: {
    currentTrainInput(){
      setTimeout(this.sendRequest, 500);
    }
  },
  created() {
    this.currentDate = this.$ds.date;
  },
  methods: {
    setCurrentTrainNumber(input: string) {
      if (!isNaN(+input)) {
        this.currentTrainInput = +input;
      }
    },
    sendRequest(){
      this.contentLoadingState = LoadingState.Loading;
      if(this.currentTrainInput !== -1){
        this.$postService.getTrainGuessResponse(this.currentDate.valueOf() / 1000, this.currentTrainInput).then((resp) => {
          this.trainGuesses = resp.trips;
          this.contentLoadingState = LoadingState.Loaded;
        });
        this.areGuessesDisplayed = true;
      }
    },
    setTimeToDisplay(value: number): string{
      return this.$ds.getTimeString(value * 1000);
    },
    setDateToDisplay(value: number): string{
      let d = new Date(value * 1000);
      return d.toLocaleString(this.$ts.availableLocales[0], { month: '2-digit', day: '2-digit' }).slice(0, 5);
    },
    goToFirstStation(trip: Trips){
      let temp: DataToRouter = { name: trip.first_station.name, id: trip.first_station.id, time: trip.trip_info.id.time}
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let t = temp as { [key: string]: any };
      this.$router.push({
        name: 'StationTimeTableFromTrainSearch',
        params: t
      })
    },
    setNewDate(date: Date){
      this.currentDate = new Date(date.getFullYear(), date.getMonth(), date.getDate(),
                                  this.currentDate.getHours(), this.currentDate.getMinutes());
    },
    checkDay(): boolean {
      if(this.$ds.date.getDate() !== this.currentDate.getDate()) { return true; }
      else { return false; }
    },
    setNewTime(time: Date){
      this.currentDate = new Date(this.currentDate.getFullYear(), this.currentDate.getMonth(), this.currentDate.getDate(),
                                  time.getHours(), time.getMinutes());
    },
  }
});

interface DataToRouter {
  name: string,
  id: string,
  time: number
}
</script>
