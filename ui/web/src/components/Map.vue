<template>
  <div class="map-container">
    <div id="map-background" class="mapboxgl-map"></div>
    <div id="map-foreground" class="mapboxgl-map"></div>
    <!-- <div class="railviz-tooltip hidden" style=""></div> -->
    <div class="map-bottom-overlay">
      <div class="sim-time-overlay">
        <div class="permalink" title="Permalink">
          <a href="#/railviz/50.65753/9.479082/6/0/0/1644346924"><i class="icon">link</i></a>
        </div>
        <button
          class="sim-icon"
          title="Simulationsmodus aktiv"
          v-show="isSimulationEnabled"
          @click="$emit('openSimWindow')">
          <i class="icon">warning</i>
        </button>
        <button class="time" id="sim-time-overlay" @click="$emit('openSimWindow')">
          {{ $ds.getTimeString(undefined, true) }}
        </button>
      </div>
      <div class="train-color-picker-overlay">
        <div>
          <input type="radio" id="train-color-picker-none" name="train-color-picker" />
          <label for="train-color-picker-none">Keine Züge</label>
        </div>
        <div>
          <input
            type="radio"
            id="train-color-picker-class"
            name="train-color-picker"
            checked />
          <label for="train-color-picker-class">Nach Kategorie</label>
        </div>
        <div>
          <input type="radio" id="train-color-picker-delay" name="train-color-picker" />
          <label for="train-color-picker-delay">Nach Verspätung</label>
        </div>
      </div>
    </div>
    <div class="railviz-contextmenu hidden" style="top: 286px; left: 760px;">
      <div class="item">
        Routen von hier
      </div>
      <div class="item">
        Routen hierher
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'

export default defineComponent({
  name: "Map",
  props: {
    isSimulationEnabled: Boolean
  },
  emits: [
    'openSimWindow'
  ],
})
</script>
