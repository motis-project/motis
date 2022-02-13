<template>
  <g :class="`part train-class-${transportClass} acc-0`">
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
    }
  },
  emits: [
    "mouseEnter",
    "mouseLeave"
  ],
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
    },
  },
  methods: {
    onMouseEnter() {
      this.$emit("mouseEnter", {
        x: this.lineStart > 95 ? 95 : this.lineStart,
        transport: this.move.move
      })
    },
    onMouseLeave() {
      this.$emit("mouseLeave");
    }
  }
})
</script>
