<template>
  <div class="app">
    <Map @openSimWindow="simWindowOpened = !simWindowOpened" :isSimulationEnabled="isSimulationEnabled"></Map>
    <LeftMenu @searchHidden="searchFieldHidden = $event"></LeftMenu>
    <div id="station-search" :class="['', searchFieldHidden ? 'overlay-hidden' : '']">
      <InputField
        iconType="place"
        :showLabel="false"
        :showAutocomplete="true"
        :isTimeCalendarField="false"
        @autocompleteElementClicked="goToTimetable"></InputField>
    </div>
    <div class="sim-time-picker-container" v-if="simWindowOpened">
      <div class="sim-time-picker-overlay">
        <div class="title">
          <input
            id="sim-mode-checkbox"
            type="checkbox"
            name="sim-mode-checkbox"
            @click="startDisableSimulation()"
            :checked="isSimulationEnabled" />
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
import Map from "../components/Map.vue"

export default defineComponent({
  name: "DesktopMain",
  components: {
    LeftMenu,
    InputField,
    Calendar,
    TimeInputField,
    Map
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
      else {
        this.$mapService.mapFlyTo( {
          mapId: "map",
          lng: element.pos.lng,
          lat: element.pos.lat,
          animate: true } );
      }
    },
    startDisableSimulation() {
      if(this.isSimulationEnabled) {
        this.cachedSimulationTime = this.$ds.date.setSeconds(0).valueOf();
        this.$ds.dateTime = new Date().valueOf();
      }
      else {
        this.$ds.dateTime = this.cachedSimulationTime;
      }
      this.isSimulationEnabled = !this.isSimulationEnabled;
    },
    changeDate(newDate: Date) {
      this.$ds.dateTime = this.formatSimTime(this.$ds.dateTime, newDate, "date");
    },
    changeTime(newTime: Date) {
      this.$ds.dateTime = this.formatSimTime(this.$ds.dateTime, newTime, "time");
    },
    formatSimTime(time: number, newTime: Date, option: ("date" | "time")): number {
      let t: Date = new Date(time);
      t.setSeconds(0);
      if(option === "date") {
        t.setDate(newTime.getDate());
        t.setMonth(newTime.getMonth());
        t.setFullYear(newTime.getFullYear());
      }
      else if(option === "time") {
        t.setHours(newTime.getHours());
        t.setMinutes(newTime.getMinutes());
      }
      else {
        return -1;
      }
      return t.valueOf();
    }
  }
});
</script>
