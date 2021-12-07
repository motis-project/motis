<template>
  <div>
    <div>
      <div v-if="showLabel" class="label">{{ labelName }}</div>
      <div class="gb-input-group">
        <div class="gb-input-icon">
          <i class="icon">{{ iconType }}</i>
        </div>
        <input
          class="gb-input"
          tabindex="1"
          @input="$emit('inputChanged', $event.target.value), onInput()"
          v-model="inputValue"
          @focus="$emit('focus', $event), onInput()"
          @blur="$emit('blur', $event), showStationAddress = false"
        />
        <div class="gb-input-widget" v-if="showArrows">
          <div class="day-buttons" >
            <div @mouseup="$emit('decreaseClick')" @mousedown="$emit('decreaseMouseDown')">
              <a class="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select"
                ><i class="icon">chevron_left</i></a
              >
            </div>
            <div @mouseup="$emit('increaseClick')" @mousedown="$emit('increaseMouseDown')">
              <a class="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select"
                ><i class="icon">chevron_right</i></a
              >
            </div>
          </div>
        </div>

      </div>
    </div>
    <StationAddressAutocomplete :input="autocompleteInput" v-show="showStationAddress" v-if="showAutocomplete" @elementClicked="onElementClicked"> </StationAddressAutocomplete>
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
  components: { StationAddressAutocomplete },
  name: "InputField",
  data() {
    return {
      showStationAddress: false,
      autocompleteInput: "",
      inputValue: ""
    }
  },
  props: {
    labelName: String,
    iconType: String,
    showLabel: Boolean,
    initInputText: String,
    showArrows: Boolean,
    showAutocomplete: Boolean,
  },
  created() {
    this.inputValue = this.initInputText ? this.initInputText : '';
  },
  methods: {
    onInput(){
      if(this.inputValue.length > 2){
        this.showStationAddress = true
        this.autocompleteInput = this.inputValue
      }
      else{
        this.showStationAddress = false
      }
    },
    onElementClicked(element: AddressGuess | StationGuess) {
      this.inputValue = element.name;
      this.autocompleteInput = element.name;
      this.showStationAddress = false;
    }
  },
});
</script>

<style>
</style>
