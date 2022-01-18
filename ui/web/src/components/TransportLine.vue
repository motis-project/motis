<template>
  <g :class="`part train-class-${transportClass} acc-0`">
    <line
      :x1="lineX1"
      y1="12"
      :x2="lineX2"
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
      :x="lineX1"
      y="40"
      text-anchor="start"
      class="train-name">
      {{ text }}
    </text>
    <rect
      :x="lineX1"
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
import Stop from "../models/Stop"

export default defineComponent({
  name: "TransportLine",
  mixins: [ ClassZConverter ],
  props: {
    move: {
      type: Object as PropType<Move>,
      required: true
    },
    allStops: {
      type: Object as PropType<Stop[]>,
      required: true
    },
    lineIndex: {
      type: Number as PropType<number>,
      required: true
    },
    lineCount: {
      type: Number as PropType<number>,
      required: true
    }
  },
  emits: [
    "mouseEnter",
    "mouseLeave"
  ],
  data() {
    return {
      size: 323,
      minWidth: 26,
      start: 0,
      end: 0,
      overallDuration: 0,
      minX1: 0,
    }
  },
  computed: {
    startPart() {
      return this.start / this.overallDuration;
    },
    endPart() {
      return this.end / this.overallDuration;
    },
    lineX1() {
      let res = this.size * this.startPart;
      if(res < this.minX1) {
        res = this.minX1;
      }
      if(res > this.size - this.minWidth * (this.lineCount - this.lineIndex)) {
        res = this.size - this.minWidth * (this.lineCount - this.lineIndex);
      }
      return res;
    },
    x2() {
      let res = this.size * this.endPart;
      const minX2 = this.lineX1 + this.minWidth;
      if(res < minX2) {
        res = minX2;
      }
      if(res > this.size) {
        res = this.size;
      }
      return res;
    },
    rectWidth() {
      return this.x2 - this.lineX1;
    },
    lineX2() {
      return this.x2 + 3;
    },
    circleX() {
      return this.lineX1 + 12;
    },
    iconX() {
      return this.lineX1 + 4;
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
    },
  },
  created() {
    this.minX1 = this.lineIndex * this.minWidth
    const overallStart = this.allStops[0].departure.time;
    this.start = this.allStops[this.move.move.range.from].departure.time - overallStart;
    if(this.move.move.range.to === this.allStops.length - 1) {
      this.end = this.allStops[this.move.move.range.to].arrival.time - overallStart
    }
    else {
      this.end = this.allStops[this.move.move.range.to].departure.time - overallStart
    }
    this.overallDuration = this.allStops[this.allStops.length - 1].arrival.time - overallStart;
  },
  methods: {
    onMouseEnter() {
      this.$emit("mouseEnter", {
        x: this.lineX1 > 95 ? 95 : this.lineX1,
        transport: this.move.move
      })
    },
    onMouseLeave() {
      this.$emit("mouseLeave");
    }
  }
})
</script>
