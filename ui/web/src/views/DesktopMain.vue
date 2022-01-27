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
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import LeftMenu from "../components/LeftMenu.vue";
import InputField from "../components/InputField.vue";
import AddressGuess from "../models/AddressGuess";
import StationGuess from "../models/StationGuess";

export default defineComponent({
  name: "DesktopMain",
  components: {
    LeftMenu,
    InputField,
  },
  data() {
    return {
      searchFieldHidden: false,
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
