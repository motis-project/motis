<template>
  <div class="overlay-container">
    <div class="overlay">
      <div id="overlay-content">
        <div id="search">
          <div class="pure-g-gutters">
            <InputField labelName="Start" iconType="place" :showLabel="true" id="start" 
              class="pure-u-1 pure-u-sm-12-24 from-location" @getStartInput="setStartInput" />
          </div>

          <div class="swap-locations-btn">
            <label class="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select">
              <input type="checkbox" @click="swapStartDest" />
              <i class="icon">swap_vert</i>
            </label>
          </div>

          <div class="pure-g-gutters">
            <InputField labelName="Ziel" iconType="place" :showLabel="true" id="dest"
              class="pure-u-1 pure-u-sm-12-24 to-location" @getDestInput="setDestInput" />
          
            <InputField labelName="Uhrzeit" iconType="schedule" :showLabel="true" id="time"
              class="pure-u-1 pure-u-1 pure-u-sm-9-24"  />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import InputField from "../components/InputField.vue";

export default defineComponent({
  name: "LeftMenu",
  components: {
    InputField,
  },
  data() {
    return {
      start: '',
      destination: '',
    }
  },
  methods: {
    swapStartDest() {
      (<HTMLInputElement>document.getElementById("dest")).value = this.start;
      (<HTMLInputElement>document.getElementById("start")).value = this.destination;
      let temp: string = this.start;
      this.start = this.destination;
      this.destination = temp;
    },
    setStartInput( input:string ){
      this.start = input;
    },
    setDestInput( input:string ){
      this.destination = input;
    },
  },
});
</script>

<style>
</style>
