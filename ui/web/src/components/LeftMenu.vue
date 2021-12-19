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
        <div class="sub-overlay-close" @click="$router.push('/')"><i class="icon">close</i></div>
      </div>
    </div>
    <div class="overlay-tabs">
      <div class="overlay-toggle">
        <i class="icon" v-on:click="(isOverlayHidden = !isOverlayHidden), $emit('searchHidden')">arrow_drop_down</i>
      </div>
      <div :class="['trip-search-toggle', isTrainSubOverlayOpened ? 'enabled' : '']">
        <i class="icon" @click="openCloseTrainSearch">train</i>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import { SubOverlayNames } from "../router/index";
import { RouteLocationNormalized } from "vue-router";

export default defineComponent({
  name: "LeftMenu",
  data() {
    return {
      isOverlayHidden: false,
      isSubOverlayHidden: true,
    };
  },
  computed: {
    isTrainSubOverlayOpened: function (): boolean {
      return this.$route.name === "TrainSearch";
    },
  },
  methods: {
    openCloseTrainSearch() {
      if (!this.isTrainSubOverlayOpened) {
        this.$router.push("/trips");
      } 
      else {
        this.$router.push("/");
      }
    },
  },
  watch: {
    $route(to: RouteLocationNormalized, from: RouteLocationNormalized) {
      if (!to.name) {
        return;
      }
      if (SubOverlayNames.includes(to.name.toString())) {
        this.isSubOverlayHidden = false;
        this.isOverlayHidden = false;
      } 
      else {
        this.isSubOverlayHidden = true;
      }
    },
  },
});
</script>

<style>
</style>
