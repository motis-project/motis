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
            @inputChanged="setCurrentTrainNumber"
          ></InputField>
        </div>
        <div class="pure-g gutters">
          <Calendar class="pure-u-1 pure-u-sm-12-24 to-location" @dateChanged="setNewDate"></Calendar>
          <TimeInputField @timeChanged="setNewTime"></TimeInputField>
        </div>
      </div>
    </div>
    <div class="trips">
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

export default defineComponent({
  name: "TrainSearch",
  components: {
    Calendar,
    TimeInputField,
    InputField,
    TransportTypeBox
  },
  data() {
    return {
      currentTrainInput: -1 as number,
      areGuessesDisplayed: false as boolean,
      currentDate: Date.now() as number,
      trainGuesses: [] as Trips[],
      dateFromCalendar: NaN as number,
      timeFromTimeField: "" as string
    };
  },
  created() {
    let d = new Date();
    this.dateFromCalendar = new Date(2020, 10, 19).valueOf() / 1000;
    this.timeFromTimeField = d.getHours() + ":" + ("0" + d.getMinutes()).slice(-2);
    this.setupDate();
  },
  methods: {
    setCurrentTrainNumber(input: string) {
      if (!isNaN(+input)) {
        this.currentTrainInput = +input;
        this.sendRequest();
      }
    },
    sendRequest(){
      if(this.currentTrainInput != -1){
        this.$postService.getTrainGuessResponse(this.currentDate, this.currentTrainInput).then((resp) => 
                                                                  (this.trainGuesses = resp.trips));
        this.areGuessesDisplayed = true;
      }
    },
    setTimeToDisplay(value: number): string{
      let d = new Date(value * 1000);
      let result: string = "";
      d.getHours() < 10 ? result += '0' + d.getHours() : result += d.getHours();
      d.getMinutes() < 10 ? result += ':0' + d.getMinutes() : result += ':' + d.getMinutes();
      return result;
    },
    setDateToDisplay(value: number): string{
      let d = new Date(value * 1000);
      let result: string = "";
      result += d.getDate() + "." + (d.getMonth() + 1);
      return result;
    },
    goToFirstStation(trip: Trips){
      let temp: DataToRouter = { name: trip.first_station.name, id: trip.first_station.id, time: trip.trip_info.id.time}
      let {sos, ...t} = temp as { [key: string]: any };
      this.$router.push({
        name: 'StationTimeTableFromTrainSearch',
        params: t
      })
    },
    setNewDate(date: Date){
      this.dateFromCalendar = date.valueOf() / 1000;
      this.setupDate();
    },
    checkDay(date: number): boolean {
      if(new Date(date * 1000).getDate() !== new Date(this.currentDate * 1000).getDate())
        return true;
      else
        return false;
    },
    setNewTime(time: string){
      this.timeFromTimeField = time;
      this.setupDate()
    },
    setupDate(){
      if((this.timeFromTimeField.length === 5 || this.timeFromTimeField.length === 4) && this.timeFromTimeField[2] === ":"){
        let inputArray: string[] = this.timeFromTimeField.split(":");
        let hours: number = +(inputArray[0]);
        let minutes: number = +(inputArray[1]);
        if(hours < 24 && hours >= 0 && minutes < 60 && minutes >= 0){
          this.currentDate = this.dateFromCalendar + (hours * 60 * 60) + (minutes * 60);
        }
      }
    }
  }
});

interface DataToRouter {
  name: string,
  id: string,
  time: number
}
</script>
