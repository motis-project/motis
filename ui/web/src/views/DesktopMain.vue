<template>
  <div class="app">
    <LeftMenu @searchHidden="searchFieldHidden = $event"></LeftMenu>
    <div id="station-search" :class="['', searchFieldHidden ? 'overlay-hidden' : '']">
      <InputField
        iconType="place"
        :showLabel="false"
        :showAutocomplete="true"
        :isTimeCalendarField="false"
        @autocompleteElementClicked="goToTimetable"></InputField>
    </div>
    <button class="sim-overlay-opener" @mousedown="simWindowOpened ? simWindowOpened = false : simWindowOpened = true">
      {{ $ds.getTimeString(undefined, true) }}
    </button>
    <div class="sim-time-picker-container" v-if="simWindowOpened">
      <div class="sim-time-picker-overlay">
        <div class="title">
          <input
            id="sim-mode-checkbox"
            type="checkbox"
            name="sim-mode-checkbox"
            @click="startDisableSimulation()"
            checked />
          <label for="sim-mode-checkbox">Simulationsmodus</label>
        </div>
        <Calendar :class="[isSimulationEnabled ? 'date' : 'date disabled']" @dateChanged="changeDate"></Calendar>
        <TimeInputField :class="[isSimulationEnabled ? 'time' : 'time disabled']" @timeChanged="changeTime"></TimeInputField>
        <button class="close" @mousedown="simWindowOpened=false">
          <i class="icon">close</i>
        </button>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import LeftMenu from "../components/LeftMenu.vue";
import InputField from "../components/InputField.vue";
import AddressGuess from "../models/AddressGuess";
import StationGuess from "../models/StationGuess";
import Calendar from "../components/Calendar.vue";
import TimeInputField from "../components/TimeInputField.vue"

export default defineComponent({
  name: "DesktopMain",
  components: {
    LeftMenu,
    InputField,
    Calendar,
    TimeInputField
  },
  data() {
    return {
      searchFieldHidden: false,
      simWindowOpened: false,
      isSimulationEnabled: true,
      cachedSimulationTime: 0,
    };
  },
  methods: {
    isStation(element: AddressGuess | StationGuess): element is StationGuess {
      return 'id' in element;
    },
    goToTimetable(element: AddressGuess | StationGuess) {
      if (this.isStation(element)) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let t = element as { [key: string]: any };
        this.$router.push({
          name: "StationTimetable",
          params: t,
        });
      }
    },
    startDisableSimulation() {
      if(this.isSimulationEnabled) {
        this.cachedSimulationTime = this.formatSimTime(this.cachedSimulationTime, this.$ds.date, "time");
        this.$ds.dateTime = new Date().valueOf();
      }
      else {
        this.$ds.dateTime = this.cachedSimulationTime;
      }
      this.isSimulationEnabled ? this.isSimulationEnabled = false : this.isSimulationEnabled = true;
    },
    changeDate(newDate: Date) {
      this.$ds.dateTime = this.formatSimTime(this.$ds.dateTime, newDate, "date");
      this.cachedSimulationTime = this.formatSimTime(this.cachedSimulationTime, newDate, "date");
    },
    changeTime(newTime: Date) {
      this.$ds.dateTime = this.formatSimTime(this.$ds.dateTime, newTime, "time");
      this.cachedSimulationTime = this.formatSimTime(this.cachedSimulationTime, newTime, "time");
    },
    formatSimTime(time: number, newTime: Date, option: ("time" | "date")): number {
      let t: Date = new Date(time);
      if(option === "time") {
        t.setHours(newTime.getHours());
        t.setMinutes(newTime.getMinutes());
        t.setSeconds(0);
      }
      else if(option === "date") {
        t.setDate(newTime.getDate());
        t.setMonth(newTime.getMonth());
        t.setFullYear(newTime.getFullYear());
      }
      else {
        return -1;
      }
      return t.valueOf();
    }
  }
});
</script>

<style>
.sim-overlay-opener {
  height: 20px;
  width: 145px;
  position: fixed;
  bottom: 20px;
  right: 150px;
  cursor: pointer;
  color: gray;
  border-color: gray;
  border-radius: 5px;
  background-color: white;
  font-family: 'Roboto', sans-serif;
}

.sim-overlay-opener:hover {
  color: white;
  background-color: gray;
}
</style>
