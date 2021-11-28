<template>
  <div class="option">
    <div class="label">Maximale Dauer (Minuten)</div>
    <div class="numeric slider control">
      <input v-model="sliderValue" type="range" min="0" max="30" step="1" />
      <input v-bind:value="sliderValue" type="text" />
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";

export default defineComponent({
  name: "InputField",
  data() {
    return {
      sliderValue: 15,
    };
  },
});
</script>
