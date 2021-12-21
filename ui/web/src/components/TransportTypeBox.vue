<template>
  <div @click="goToTripDetails()">
    <div
      :class="`train-box train-class-${transport.clasz} with-tooltip`"
      :data-tooltip="
        $t.provider + ': ' + transport.provider + (transport.train_nr ? `\n${$t.trainNr}: ` + transport.train_nr : '')
      ">
      <svg class="train-icon"><use :xlink:href="'#' + icon"></use></svg>
      <span class="train-name">{{ transport.name }}</span>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, PropType } from "vue";
import Transport from "../models/Transport";
import Trip from "../models/Trip";

export default defineComponent({
  name: "TransportTypeBox",
  props: {
    transport: {
      type: Object as PropType<Transport>,
      required: true,
    },
    trip: {
      type: Object as PropType<Trip>,
      required: false,
    },
  },
  computed: {
    icon: function (): string {
      switch (this.transport.clasz) {
      case 1:
      case 2:
      case 4:
      case 5:
      case 6:
        return "train";
      case 0:
        return "plane";
      case 7:
        return "sbahn";
      case 8:
        return "ubahn";
      case 9:
        return "tram";
      case 11:
        return "ship";
      case 3:
      default:
        return "bus";
      }
    },
  },
  methods: {
    goToTripDetails() {
      if (this.trip) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let t = this.trip as { [key: string]: any };
        this.$router.push({
          name: "Trip",
          params: t,
        });
      }
    },
  },
});
</script>
