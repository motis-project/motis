<template>
  <div>
    <div class="paper" v-show="stationGuesses.length > 0 || addressGuesses.length > 0">
      <ul class="proposals">
        <li
          @mouseenter="focusElement(gIndex)"
          :class="['', g.isMouseOver ? 'selected' : '']"
          v-for="(g, gIndex) in stationGuesses"
          :key="g.stationGuess.id"
          :title="g.stationGuess.name"
          @mousedown="$emit('elementClicked', g.stationGuess)"
        >
          <i class="icon">train</i>
          <span class="station"> {{ g.stationGuess.name }} </span>
        </li>
        <li
          @mouseenter="focusElement(gIndex + stationGuesses.length)"
          :class="['', g.isMouseOver ? 'selected' : '']"
          v-for="(g, gIndex) in addressGuesses"
          :key="g.addressGuess.id"
          :title="g.addressGuess.name + ', ' + g.region.split(', ')[0]"
          @mousedown="$emit('elementClicked', g.addressGuess)"
        >
          <i class="icon">place</i>
          <span class="address-name">{{ g.addressGuess.name }}</span>
          <span class="address-region"> {{ g.region }}</span>
        </li>
      </ul>
    </div>
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
      currentFocused: 0
    };
  },
  methods: {
    getRegion(g: AddressGuess) {
      let r6 = g.regions.filter((r) => r.admin_level == 6);
      let res = "";
      if (r6.length != 0) {
        res += r6[0].name;
      }

      let r4 = g.regions.filter((r) => r.admin_level == 4);
      if (r4.length != 0 && r6.length == 0) {
        res += r4[0].name;
      }

      let r2 = g.regions.filter((r) => r.admin_level == 2);
      if (r2.length != 0) {
        if (res !== "") {
          res += ", ";
        }
        res += r2[0].name;
      }
      return res;
    },
    keyDown(e: KeyboardEvent) {
      if(this.currentFocused === -1) {
        return;
      }
      switch(e.key) {
        case 'ArrowUp':
          if(this.currentFocused > 0) {
            this.focusElement(this.currentFocused - 1);
          }
          else {
            this.focusElement(this.stationGuesses.length + this.addressGuesses.length - 1);
          }
          break;
        case 'ArrowDown':
          if(this.currentFocused < this.stationGuesses.length + this.addressGuesses.length - 1) {
            this.focusElement(this.currentFocused + 1);
          }
          else {
            this.focusElement(0);
          }
          break;
        case 'Enter':
          let element = this.getElement(this.currentFocused);
          if(element) {
            this.$emit('elementClicked', ("addressGuess" in element ? element.addressGuess : element.stationGuess));
          }
          break;
      }
    },
    focusElement(index: number) {
      for(let i of this.stationGuesses) {
        i.isMouseOver = false;
      }
      for(let i of this.addressGuesses) {
        i.isMouseOver = false;
      }

      let element = this.getElement(index);
      if(element) {
        element.isMouseOver = true;
        this.currentFocused = index;
      }
      else {
        this.currentFocused = -1;
      }
    },
    getElement(index: number) {
      if(index > this.stationGuesses.length - 1 && this.addressGuesses.length > 0) {
        index -= this.stationGuesses.length;
        return this.addressGuesses[index];
      }
      else if(this.stationGuesses.length > 0) {
        return this.stationGuesses[index];
      }
    }
  },
  created() {
    window.addEventListener('keydown', this.keyDown);
  },
  props: {
    input: String,
  },
  watch: {
    input(newInput) {
      this.$postService.getStationGuessResponse(newInput, 6).then(
        (data) => {
          this.stationGuesses = data.guesses.map<Station>((g) => {
            return {
              stationGuess: g,
              isMouseOver: false,
            };
          });
          this.focusElement(0);
        }
      );
      this.$postService.getAddressGuessResponse(newInput).then(
        (data) => {
          this.addressGuesses = data.guesses.splice(0, 9).map<Address>((g) => {
            return {
              addressGuess: g,
              region: this.getRegion(g),
              isMouseOver: false,
            };
          });
          this.focusElement(0);
        }
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

