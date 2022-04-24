<template>
  <g :class="['part', `train-class-${transportClass}`, 'acc-0', highlightClass]">
    <line
      :x1="lineStart"
      y1="12"
      :x2="lineEnd"
      y2="12"
      class="train-line"></line>
    <circle
      :cx="circleX"
      cy="12"
      r="12"
      class="train-circle"></circle>
    <use
      :xlink:href="'#' + transportIcon"
      class="train-icon"
      :x="iconX"
      y="4"
      width="16"
      height="16"></use>
    <text
      :x="lineStart"
      y="40"
      text-anchor="start"
      class="train-name">
      {{ text }}
    </text>
    <rect
      :x="lineStart"
      y="0"
      :width="rectWidth"
      height="24"
      class="tooltipTrigger"
      @mouseenter="onMouseEnter"
      @mouseleave="onMouseLeave"></rect>
  </g>
</template>


<script lang="ts">
import { defineComponent, PropType } from 'vue'
import { Move } from "../models/TripResponseContent"
import ClassZConverter from "../mixins/ClassZConverter"
import { MapHoverOptions } from "../views/ConnectionSearch.vue"
import Stop from '../models/Stop'

export default defineComponent({
  name: "TransportLine",
  mixins: [ ClassZConverter ],
  props: {
    move: {
      type: Object as PropType<Move>,
      required: true,
    },
    lineStart: {
      type: Number as PropType<number>,
      required: true
    },
    lineEnd: {
      type: Number as PropType<number>,
      required: true
    },
    connectionIndex: {
      type: Number as PropType<number>,
      required: true
    },
    stops: {
      type: Array as PropType<Stop[]>,
      required: false
    },
    mapHoverOptions: {
      type: Object as PropType<MapHoverOptions>,
      required: false,
      default: null
    }
  },
  emits: [
    "mouseEnter",
    "mouseLeave"
  ],
  data() {
    return {
      highlightClass: ""
    }
  },
  computed: {
    rectWidth() {
      return this.lineEnd - this.lineStart;
    },
    lineX2() {
      return this.lineEnd + 3;
    },
    circleX() {
      return this.lineStart + 12;
    },
    iconX() {
      return this.lineStart + 4;
    },
    text() {
      if("name" in this.move.move) {
        return this.move.move.name;
      }
      return "";
    },
    transportClass() {
      if("clasz" in this.move.move) {
        return this.move.move.clasz.toString();
      }
      else if(this.move.move.mumo_type === "foot") {
        return "walk";
      }
      else {
        return this.move.move.mumo_type;
      }
    },
    transportIcon() {
      if("clasz" in this.move.move) {
        return this.convertClassZ(this.move.move.clasz);
      }
      else {
        return this.move.move.mumo_type;
      }
    }
  },
  watch: {
    mapHoverOptions() {
      this.highlightClass = this.getHighlightClass();
    }
  },
  methods: {
    onMouseEnter() {
      this.$mapService.mapHighlightConnections([this.connectionIndex]);
      this.$emit("mouseEnter", {
        x: this.lineStart > 95 ? 95 : this.lineStart,
        transport: this.move.move
      })
    },
    onMouseLeave() {
      this.$mapService.mapHighlightConnections([]);
      this.$emit("mouseLeave");
    },
    getHighlightClass() {
      if(!this.mapHoverOptions || !this.mapHoverOptions.connectionIds.includes(this.connectionIndex) || !this.stops) {
        this.$emit("mouseLeave");
        return "";
      }
      const stopIds = this.stops.map(s => s.station.id);
      if((this.mapHoverOptions.stationIds.some(s => stopIds.includes(s.departure) && stopIds.includes(s.arrival)))
        || (this.stops[0].station === this.mapHoverOptions.departureStation && this.stops[this.stops.length - 1].station === this.mapHoverOptions.arrivalStation)) {
        this.$emit("mouseEnter", {
          x: this.lineStart > 95 ? 95 : this.lineStart,
          transport: this.move.move
        })
        return "highlighted";
      }
      return "faded";
    }
  }
})
</script>
