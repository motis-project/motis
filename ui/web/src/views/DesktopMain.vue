<template>
  <div class="app">
    <div class="map-container">
      <LeftMenu @searchHidden="searchFieldHidden = !searchFieldHidden"></LeftMenu>
      <div id="station-search" :class="['', searchFieldHidden ? 'overlay-hidden' : '']">
          <InputField labelName="Stationsearch" iconType="place" :showLabel="false" />
            <div class="paper hide">
              <ul class="proposals"></ul>
            </div>
        </div>
      </div>
    </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import LeftMenu from '../components/LeftMenu.vue'
import InputField from '../components/InputField.vue'

export default defineComponent({
  name: 'DesktopMain',
  components:
  {
    LeftMenu,
    InputField
  },
  data() {
    return {
      searchFieldHidden: false
    }
  }
});
</script>

<style>
</style>
