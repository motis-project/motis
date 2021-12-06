<template>
  <div :class="['overlay-container', isOverlayHidden ? 'hidden' : '']">
    <div class="overlay">
      <div id="overlay-content">
        <ConnectionSearchOverlay></ConnectionSearchOverlay>
      </div>
      <div :class="['sub-overlay', isSubOverlayHidden ? 'hidden' : '']">
        <div id="sub-overlay-content">
          <TrainSearch></TrainSearch>
        </div>
        <div class="sub-overlay-close" @click="isSubOverlayHidden = true"><i class="icon">close</i></div>
      </div>
    </div>
    <div class="overlay-tabs">
      <div class="overlay-toggle">
        <i class="icon" v-on:click="(isOverlayHidden = !isOverlayHidden), $emit('searchHidden')">arrow_drop_down</i>
      </div>
      <div :class="['trip-search-toggle', isSubOverlayHidden ? '' : 'enabled']">
        <i class="icon" @click="isSubOverlayHidden = !isSubOverlayHidden">train</i>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import ConnectionSearchOverlay from "../views/ConnectionSearch.vue";
import TrainSearch from "../views/TrainSearch.vue";

export default defineComponent({
  name: "LeftMenu",
  components: {
    ConnectionSearchOverlay,
    TrainSearch,
  },
  data() {
    return {
      isOverlayHidden: false,
      isSubOverlayHidden: true,
    };
  },
});
</script>

<style>
</style>
