<template>
  <div class="paper">
    <ul class="proposals">
      <li
        @mouseover="onMouseOver"
        :class="['', isMouseOver ? 'selected' : '']"
        v-for="g in stationGuesses"
        :key="g.id"
        :title="g.name"
        @click="$emit('elementClicked', g)"
      >
        <i class="icon">train</i>
        <span class="station"> {{ g.name }} </span>
      </li>
      <li
        @mouseover="onMouseOver"
        :class="['', isMouseOver ? 'selected' : '']"
        v-for="g in addressGuesses"
        :key="g.addressGuess.id"
        :title="g.addressGuess.name"
        @click="$emit('elementClicked', g)"
      >
        <i class="icon">place</i>
        <span class="address-name">{{ g.addressGuess.name }}</span>
        <span class="address-region">
          {{ g.region }}</span
        >
      </li>
    </ul>
  </div>
</template>

<script lang="ts">
import AddressGuess from "@/models/AddressGuess";
import { defineComponent } from "vue";
import StationGuess from "../models/StationGuess";

export default defineComponent({
  name: "StationAddressAutocomplete",
  data() {
    return {
      isMouseOver: false,
      stationGuesses: [] as StationGuess[],
      addressGuesses: [] as Address[],
    };
  },
  methods: {
    onMouseOver() {},
    getRegion(g: AddressGuess) {
      let r6 = g.regions.filter((r) => r.admin_level == 6);
      let res = '';
      if(r6.length != 0) {
        res += r6[0].name;
      }

      let r4 = g.regions.filter((r) => r.admin_level == 4)
      if(r4.length != 0 && r6.length == 0) {
        res += r4[0].name;
      }

      let r2 = g.regions.filter((r) => r.admin_level == 2);
      if(r2.length != 0) {
        if(res !== '') {
          res += ', ';
        }
        res += r2[0].name;
      }
      return res;
    }
  },
  props: {
    input: String,
  },
  watch: {
    input(newInput) {
      this.$postService.getStationGuessResponse(newInput, 6).then((data) => (this.stationGuesses = data.guesses));
      this.$postService.getAddressGuessResponse(newInput).then(
        (data) =>
          (this.addressGuesses = data.guesses.map<Address>((g) => {
            return {
              addressGuess: g,
              region: this.getRegion(g),
              isMouseOver: false,
            };
          }))
      );
    },
  },
});
interface Address {
  addressGuess: AddressGuess;
  region: string;
  isMouseOver: boolean;
}
</script>

