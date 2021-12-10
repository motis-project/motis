<template>
  <div class="app">
    <LeftMenu @searchHidden="searchFieldHidden = !searchFieldHidden"></LeftMenu>
    <div id="station-search" :class="['', searchFieldHidden ? 'overlay-hidden' : '']">
      <InputField labelName="Stationsearch" iconType="place" :showLabel="false" :showAutocomplete="true" @autocompleteElementClicked="goToTimetable"/>
      <div class="paper hide">
        <ul class="proposals"></ul>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import LeftMenu from "../components/LeftMenu.vue";
import InputField from "../components/InputField.vue";
import AddressGuess from "@/models/AddressGuess";
import StationGuess from "@/models/StationGuess";

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
      return 'name' in element;
    },
    goToTimetable(element: AddressGuess | StationGuess) {
      if (this.isStation(element)) {
        let { pos, ...t } = element as { [key: string]: any };
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
