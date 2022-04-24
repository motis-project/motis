<template>
  <div class="option">
    <div class="label">
      {{ $t.maxDuration }}
    </div>
    <div class="numeric slider control">
      <input
        v-model="sliderValueComputed"
        type="range"
        min="0"
        max="30"
        step="1" />
      <input v-model="sliderValueComputed" type="text" />
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, PropType } from "vue";

export default defineComponent({
  name: "Slider",
  props: {
    initSliderValue: {
      type: Number as PropType<number>,
      required: true,
    }
  },
  emits: ["sliderValueChanged"],
  data() {
    return {
      sliderValue: this.initSliderValue,
    };
  },
  computed: {
    sliderValueComputed: {
      get: function() : string {
        return this.sliderValue.toString();
      },
      set: function(value: string) {
        let n = +value;
        if(value !== '' && !isNaN(n)) {
          this.sliderValue = n - 1;
          this.sliderValue = n > 30 ? 30 : (n < 0 ? 0 : n);
        }
      }
    }
  },
  watch: {
    sliderValue() {
      this.$emit("sliderValueChanged", this.sliderValue);
    }
  }
});
</script>
