<template>
  <div :class="['overlay-container', isOverlayHidden ? 'hidden' : '']">
    <div class="overlay">
      <div id="overlay-content">
        <RouterView name="overlay" v-slot="{ Component }">
          <keep-alive include="ConnectionSearch">
            <component :is="Component"></component>
          </keep-alive>
        </RouterView>
      </div>
      <div :class="['sub-overlay', isSubOverlayHidden ? 'hidden' : '']">
        <div id="sub-overlay-content">
          <RouterView name="subOverlay" v-slot="{ Component }">
            <keep-alive include="TrainSearch">
              <component :is="Component">
              </component>
            </keep-alive>
          </RouterView>
        </div>
        <div class="sub-overlay-close" @click="$router.push('/')">
          <i class="icon">close</i>
        </div>
      </div>
    </div>
    <div class="overlay-tabs">
      <div class="overlay-toggle">
        <i class="icon" @click="(isOverlayHidden = !isOverlayHidden)">arrow_drop_down</i>
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
import { RouteLocationNormalized, RouterView } from "vue-router";

export default defineComponent({
  name: "LeftMenu",
  components: {
    RouterView
  },
  emits: ["searchHidden"],
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
  watch: {
    $route(to: RouteLocationNormalized) {
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
    isOverlayHidden() {
      this.$emit('searchHidden', this.isOverlayHidden)
    }
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
});
</script>

<style>
</style>
