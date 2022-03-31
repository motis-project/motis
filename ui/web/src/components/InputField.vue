<template>
  <div>
    <div>
      <div v-if="showLabel" class="label">
        {{ labelName }}
      </div>
      <div :class="['gb-input-group', isFocused ? 'gb-input-group-selected' : '']">
        <div class="gb-input-icon">
          <i class="icon">{{ iconType }}</i>
        </div>
        <input
          :inputmode="isTimeCalendarField ? 'numeric' : ''"
          class="gb-input"
          :tabindex="tabIndex"
          @input="onInput"
          v-model="inputValue"
          @focus="$emit('focus', $event), onInput, isFocused = true"
          @blur="onBlur"
          @keydown="$emit('keydown', $event)"
          @mouseup="$emit('mouseup', $event)" />
        <div class="gb-input-widget" v-if="showArrows">
          <div class="day-buttons">
            <button
              @mouseup="$emit('decreaseClick')"
              @mousedown="$emit('decreaseMouseDown')"
              class="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select">
              <i class="icon">chevron_left</i>
            </button>
            <button
              @mouseup="$emit('increaseClick')"
              @mousedown="$emit('increaseMouseDown')"
              class="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select">
              <i class="icon">chevron_right</i>
            </button>
          </div>
        </div>
      </div>
    </div>
    <StationAddressAutocomplete
      :input="inputValue"
      :showList="showStationAddress"
      v-if="showAutocomplete"
      @elementClicked="onElementClicked">
    </StationAddressAutocomplete>
  </div>
</template>


<script lang="ts">
import { defineComponent } from "vue";
import StationAddressAutocomplete from "./StationAddressAutocomplete.vue";
import StationGuess from "../models/StationGuess";
import AddressGuess from "../models/AddressGuess";

export default defineComponent({
  name: "InputField",
  components: {
    StationAddressAutocomplete
  },
  props: {
    labelName: String,
    iconType: String,
    showLabel: Boolean,
    initInputText: String,
    showArrows: Boolean,
    showAutocomplete: Boolean,
    isTimeCalendarField: Boolean,
    tabIndex: {
      type: Number,
      requiered: false,
      default: 0
    }
  },
  emits: [
    "inputChanged",
    "focus",
    "blur",
    "decreaseClick",
    "decreaseMouseDown",
    "increaseClick",
    "increaseMouseDown",
    "autocompleteElementClicked",
    "isTimeCalendarField",
    "inputChangedNative",
    "keydown",
    "mouseup"
  ],
  data() {
    return {
      showStationAddress: false,
      inputValue: "",
      isFocused: false,
      savedInputValue: "" as (string | null)
    }
  },
  watch: {
    initInputText(newValue: string) {
      this.inputValue = newValue
    },
    isFocused() {
      if(this.isFocused) {
        this.setShowStationAddress();
      }
    }
  },
  created() {
    this.inputValue = this.initInputText ? this.initInputText : '';
  },
  methods: {
    onInput(event: Event){
      this.setShowStationAddress()
      this.$emit('inputChanged', (event.target as HTMLInputElement).value)
      this.$emit("inputChangedNative", event);
    },
    setShowStationAddress() {
      if(this.inputValue.length > 2){
        this.showStationAddress = true
      }
      else{
        this.showStationAddress = false
      }
    },
    onElementClicked(element: AddressGuess | StationGuess) {
      this.inputValue = element.name;
      this.savedInputValue = element.name;
      this.showStationAddress = false;
      this.$emit('autocompleteElementClicked', element);
    },
    onBlur(event: Event) {
      if(this.savedInputValue) {
        this.inputValue = this.savedInputValue;
        this.savedInputValue = null;
      }
      this.$emit('blur', event);
      this.showStationAddress = false;
      this.isFocused = false
    }
  },
});
</script>

<style>
</style>
