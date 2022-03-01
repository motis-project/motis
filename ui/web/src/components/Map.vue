<template>
  <div class="map-container">
    <div id="map-background" class="mapboxgl-map"></div>
    <div id="map-foreground" class="mapboxgl-map"></div>
    <div
      :class="['railviz-tooltip',
               tooltipState === TooltipState.Station ? 'station' : (tooltipState === TooltipState.Train ? 'train' : ''),
               tooltipState ? 'visible' : 'hidden']"
      :style="`top: ${tooltipPosition.y}px; left: ${tooltipPosition.x}px;`">
      <div v-if="tooltipState === TooltipState.Station" class="station-name">
        {{ tooltipStationName }}
      </div>
      <template v-else-if="tooltipState === TooltipState.Train">
        <div class="transport-name">
          {{ tooltipTransportInfo.name }}
        </div>
        <div class="departure">
          <span class="station">{{ tooltipTransportInfo.departureStation }}</span>
          <div class="time no-delay-infos">
            <span class="schedule">{{ tooltipTransportInfo.departureTime }}</span>
          </div>
        </div>
        <div class="arrival">
          <span class="station">{{ tooltipTransportInfo.arrivalStation }}</span>
          <div class="time no-delay-infos">
            <span class="schedule">{{ tooltipTransportInfo.arrivalTime }}</span>
          </div>
        </div>
      </template>
    </div>
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
          <input
            type="radio"
            id="train-color-picker-none"
            :value="RadioState.None"
            v-model="radioState" />
          <label for="train-color-picker-none">{{ $t.noTrains }}</label>
        </div>
        <div>
          <input
            type="radio"
            id="train-color-picker-class"
            :value="RadioState.Class"
            v-model="radioState" />
          <label for="train-color-picker-class">{{ $t.byCategory }}</label>
        </div>
        <div>
          <input
            type="radio"
            id="train-color-picker-delay"
            :value="RadioState.Delay"
            v-model="radioState" />
          <label for="train-color-picker-delay">{{ $t.byDelay }}</label>
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
import { MapTooltipOptions } from "../services/MOTISMapService"

export default defineComponent({
  name: "Map",
  props: {
    isSimulationEnabled: Boolean
  },
  emits: [
    'openSimWindow'
  ],
  data() {
    return {
      TooltipState: TooltipState,
      tooltipState: TooltipState.Hidden,
      tooltipPosition: {
        x: 0,
        y: 0
      },
      tooltipStationName: "",
      tooltipTransportInfo: {} as TooltipTransportInfo,
      RadioState: RadioState,
      radioState: RadioState.Class
    }
  },
  watch: {
    radioState() {
      console.log(this.radioState);
      this.$mapService.mapShowTrains(this.radioState !== RadioState.None);
      if(this.radioState !== RadioState.None) {
        this.$mapService.mapUseTrainClassColors(this.radioState === RadioState.Class);
      }
    }
  },
  created() {
    this.$mapService.mapSetTooltipDelegates.push(this.mapTrainOrStationHovered);
  },
  methods: {
    mapTrainOrStationHovered(options: MapTooltipOptions) {
      this.tooltipState = options.hoveredTrain ? TooltipState.Train : (options.hoveredStation ? TooltipState.Station : TooltipState.Hidden);
      if(this.tooltipState === TooltipState.Hidden) {
        return;
      }

      this.tooltipPosition = {
        x: options.mouseX - 120,
        y: options.mouseY + 20
      }
      if(options.hoveredTrain) {
        this.tooltipTransportInfo = {
          name: options.hoveredTrain.names[0],
          departureStation: options.hoveredTrain.departureStation,
          departureTime: this.$ds.getTimeString(options.hoveredTrain.departureTime),
          arrivalStation: options.hoveredTrain.arrivalStation,
          arrivalTime: this.$ds.getTimeString(options.hoveredTrain.arrivalTime)
        };
      }
      else if(options.hoveredStation) {
        this.tooltipStationName = options.hoveredStation;
      }
    },
  }
})

interface TooltipTransportInfo {
  name: string,
  departureStation: string,
  departureTime: string,
  arrivalStation: string,
  arrivalTime: string
}

enum TooltipState {
  Hidden,
  Train,
  Station
}

enum RadioState {
  None,
  Class,
  Delay
}
</script>
