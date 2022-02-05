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
    <div class="sim-time-picker-container" v-if="simWindowOpened">
      <div class="sim-time-picker-overlay">
        <div class="title">
          <input id="sim-mode-checkbox" type="checkbox" name="sim-mode-checkbox" />
          <label for="sim-mode-checkbox">Simulationsmodus</label>
        </div>
        <Calendar class="date"></Calendar>
        <TimeInputField class="time"></TimeInputField>
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
      simWindowOpened: true,
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
    }
  }
});
</script>

<style>
</style>
