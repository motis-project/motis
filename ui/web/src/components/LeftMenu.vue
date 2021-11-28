<template>
  <div class="overlay-container">
    <div class="overlay">
      <div id="overlay-content">
        <div id="search">
          <div class="pure-g-gutters">
            <InputField labelName="Start" iconType="place" :showLabel="true" :initInputText="start"
              class="pure-u-1 pure-u-sm-12-24 from-location" @inputChanged="setStartInput" />
          </div>

          <div class="swap-locations-btn">
            <label class="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select">
              <input type="checkbox" @click="swapStartDest" />
              <i class="icon">swap_vert</i>
            </label>
          </div>

          <div class="pure-g-gutters">
            <InputField labelName="Ziel" iconType="place" :showLabel="true" :initInputText="destination"
              class="pure-u-1 pure-u-sm-12-24 to-location" @inputChanged="setDestInput" />
          
            <InputField labelName="Uhrzeit" iconType="schedule" :showLabel="true" :initInputText="time"
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
      time:  (new Date).getHours() + ":" + ("0" + (new Date).getMinutes()).slice(-2),
    }
  },
  methods: {
    swapStartDest() {
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
