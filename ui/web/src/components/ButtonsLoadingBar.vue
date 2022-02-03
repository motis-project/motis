<template>
  <div class="spinner">
    <div class="bounce1"></div>
    <div class="bounce2"></div>
    <div class="bounce3"></div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'

export default defineComponent({
  name: "ButtonsLoadingBar"
})

export enum UpperButtonState {
  Loading,
  Loaded
}

export enum LowerButtonState {
  Loading,
  Loaded
}
</script>


<style scoped>
.center {
    margin: auto;
}
</style>
