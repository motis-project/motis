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
          @blur="$emit('blur', $event), showStationAddress = false, isFocused = false"
          @keydown="$emit('keydown', $event)"
          @mouseup="$emit('mouseup', $event)" />
        <div class="gb-input-widget" v-if="showArrows">
          <div class="day-buttons">
            <div @mouseup="$emit('decreaseClick')" @mousedown="$emit('decreaseMouseDown')">
              <a
                class="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select"><i class="icon">chevron_left</i></a>
            </div>
            <div @mouseup="$emit('increaseClick')" @mousedown="$emit('increaseMouseDown')">
              <a
                class="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select"><i class="icon">chevron_right</i></a>
            </div>
          </div>
        </div>
      </div>
    </div>
    <StationAddressAutocomplete
      :input="inputValue"
      v-show="showStationAddress"
      v-if="showAutocomplete"
      @elementClicked="onElementClicked">
    </StationAddressAutocomplete>
  </div>
</template>


<!-- How to use this component:
    <InputField :showLabel=true labelName="Start" iconType="place"/>
    to not display the labelName set :showLabel to false
-->


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
      isFocused: false
    }
  },
  watch: {
    initInputText(newValue: string) {
      this.inputValue = newValue
    },
    inputValue(newValue: string) {
      this.$emit('inputChanged', newValue)
    }
  },
  created() {
    this.inputValue = this.initInputText ? this.initInputText : '';
  },
  methods: {
    onInput(event: Event){
      if(this.inputValue.length > 2){
        this.showStationAddress = true
      }
      else{
        this.showStationAddress = false
      }
      this.$emit("inputChangedNative", event);
    },
    onElementClicked(element: AddressGuess | StationGuess) {
      this.inputValue = element.name;
      this.showStationAddress = false;
      this.$emit('autocompleteElementClicked', element);
    },
  },
});
</script>

<style>
</style>
