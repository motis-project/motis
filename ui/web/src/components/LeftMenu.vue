<template>
  <div :class="['overlay-container', isOverlayHidden ? 'hidden' : '']">
    <div class="overlay">
      <div id="overlay-content">
        <router-view name="overlay"></router-view>
      </div>
      <div :class="['sub-overlay', isSubOverlayHidden ? 'hidden' : '']">
        <div id="sub-overlay-content">
          <router-view name="subOverlay"></router-view>
        </div>
        <div class="sub-overlay-close" @click="openCloseTrainSearch"><i class="icon">close</i></div>
      </div>
    </div>
    <div class="overlay-tabs">
      <div class="overlay-toggle">
        <i class="icon" v-on:click="(isOverlayHidden = !isOverlayHidden), $emit('searchHidden')">arrow_drop_down</i>
      </div>
      <div :class="['trip-search-toggle', isSubOverlayHidden ? '' : 'enabled']">
        <i class="icon" @click="openCloseTrainSearch">train</i>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";

export default defineComponent({
  name: "LeftMenu",
  data() {
    return {
      isOverlayHidden: false,
      isSubOverlayHidden: true,
    };
  },
  methods: {
    openCloseTrainSearch() {
      this.isSubOverlayHidden = !this.isSubOverlayHidden
      if(!this.isSubOverlayHidden) {
        this.$router.push("/trips");
      }
      else {
        this.$router.push("/");
      }
    }
  }
});
</script>

<style>
</style>
