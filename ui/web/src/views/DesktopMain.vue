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
      Open Sim Window
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
        <div class="close" @mousedown="simWindowOpened=false">
          <i class="icon">close</i>
        </div>
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
        this.$ds.cachedSimulationTime = this.$ds.simulationTime;
        this.$ds.simulationTime = new Date().valueOf();
      }
      else {
        this.$ds.simulationTime = this.$ds.cachedSimulationTime;
        this.$ds.cachedSimulationTime = 0;
      }
      this.isSimulationEnabled ? this.isSimulationEnabled = false : this.isSimulationEnabled = true;
    },
    changeDate(newDate: Date) {
      let t: Date = new Date(this.$ds.simulationTime)
      t.setDate(newDate.getDate());
      t.setMonth(newDate.getMonth());
      t.setFullYear(newDate.getFullYear());
      this.$ds.simulationTime = t.valueOf();
    },
    changeTime(newTime: Date) {
      let t: Date = new Date(this.$ds.simulationTime)
      t.setHours(newTime.getHours());
      t.setMinutes(newTime.getMinutes());
      this.$ds.simulationTime = t.valueOf();
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
