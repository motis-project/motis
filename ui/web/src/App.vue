<template>
  <desktop-main></desktop-main>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import DesktopMain from './views/DesktopMain.vue';

export default defineComponent({
  name: 'App',
  components:
  {
    DesktopMain
  }
});
</script>

<style>
  @import "../external_lib/mapbox-gl.css";
  @import "../external_lib/grid/grids-min.css";
  @import "../external_lib/grid/grids-responsive-min.css";
  @import "../external_lib/grid/grids-gutters.css";
  @import "../style/global.css";
  @import "../style/taglist.css";
  @import "../style/calendar.css";
  @import "../style/time.css";
  @import "../style/button.css";
  @import "../style/input.css";
  @import "../style/typeahead.css";
  @import "../style/spinner.css";
  @import "../style/app.css";
  @import "../style/transport-graph.css";
</style>
