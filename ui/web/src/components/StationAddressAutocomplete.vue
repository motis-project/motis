<template>
  <div class="paper">
    <ul class="proposals">
      <li
        @mouseenter="g.isMouseOver = true"
        @mouseleave="g.isMouseOver = false"
        :class="['', g.isMouseOver ? 'selected' : '']"
        v-for="g in stationGuesses"
        :key="g.stationGuess.id"
        :title="g.stationGuess.name"
        @mousedown="$emit('elementClicked', g.stationGuess)"
      >
        <i class="icon">train</i>
        <span class="station"> {{ g.stationGuess.name }} </span>
      </li>
      <li
        @mouseenter="g.isMouseOver = true"
        @mouseleave="g.isMouseOver = false"
        :class="['', g.isMouseOver ? 'selected' : '']"
        v-for="g in addressGuesses"
        :key="g.addressGuess.id"
        :title="g.addressGuess.name"
        @mousedown="$emit('elementClicked', g.addressGuess)"
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
import AddressGuess from "../models/AddressGuess";
import { defineComponent } from "vue";
import StationGuess from "../models/StationGuess";

export default defineComponent({
  name: "StationAddressAutocomplete",
  data() {
    return {
      isMouseOver: false,
      stationGuesses: [] as Station[],
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
      this.$postService.getStationGuessResponse(newInput, 6).then((data) => (this.stationGuesses = data.guesses.map<Station>((g) => {
        return{
          stationGuess: g,
          isMouseOver: false,
        }
      })));
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
interface Station {
  stationGuess: StationGuess;
  isMouseOver: boolean;
}
</script>

