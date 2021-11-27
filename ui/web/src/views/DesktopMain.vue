<template>
  <div class="app">
    <div class="map-container">
      <left-menu></left-menu>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import LeftMenu from '../components/LeftMenu.vue'

export default defineComponent({
  name: 'DesktopMain',
  components:
  {
    LeftMenu
  }
});
</script>

<style>
</style>
