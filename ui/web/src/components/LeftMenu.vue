<template>
  <div class="overlay-container">
    <div class="overlay">
      <div id="overlay-content">
        <div id="search">
          <div class="pure-g-gutters">
            <InputField labelName="Start" iconType="place" :showLabel="true" 
              class="pure-u-1 pure-u-sm-12-24 from-location"/>
          </div>

          <div class="swap-locations-btn">
              <label class=" gb-button gb-button-small gb-button-circle gb-button-outline 
                gb-button-PRIMARY_COLOR disable-select">
              <input type="checkbox">
              <i class="icon">swap_vert</i>
            </label>
          </div>
          
          <div class="pure-g-gutters">
            <InputField labelName="Ziel" iconType="place" :showLabel="true" 
              class="pure-u-1 pure-u-sm-12-24 from-location"/>
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
});
</script>

<style>
</style>
