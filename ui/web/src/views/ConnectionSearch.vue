<template>
  <div id="search">
    <div class="pure-g-gutters">
      <div class="pure-u-1 pure-u-sm-12-24 from-location">
        <InputField
          :labelName="$t.start"
          iconType="place"
          :showLabel="true"
          :initInputText="start"
          @inputChanged="setStartInput"
          :showAutocomplete="true"></InputField>

        <div class="mode-picker-btn" @click="optinsButton1Click">
          <div :class="['mode', firstOptions.foot ? 'enabled' : '']">
            <i class="icon">directions_walk</i>
          </div>
          <div :class="['mode', firstOptions.bicycle ? 'enabled' : '']">
            <i class="icon">directions_bike</i>
          </div>
          <div :class="['mode', firstOptions.car ? 'enabled' : '']">
            <i class="icon">directions_car</i>
          </div>
        </div>

        <button class="swap-locations-btn">
          <label class="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select">
            <input type="checkbox" @click="swapStartDest" />
            <i class="icon">swap_vert</i>
          </label>
        </button>
      </div>
      <Calendar class="pure-u-1 pure-u-sm-12-24"></Calendar>
    </div>

    <div class="pure-g-gutters">
      <div class="pure-u-1 pure-u-sm-12-24 from-location">
        <InputField
          :labelName="$t.destination"
          iconType="place"
          :showLabel="true"
          :initInputText="destination"
          @inputChanged="setDestInput"
          :showAutocomplete="true"></InputField>
        <div class="mode-picker-btn" @click="optinsButton2Click">
          <div :class="['mode', secondOptions.foot ? 'enabled' : '']">
            <i class="icon">directions_walk</i>
          </div>
          <div :class="['mode', secondOptions.bicycle ? 'enabled' : '']">
            <i class="icon">directions_bike</i>
          </div>
          <div :class="['mode', secondOptions.car ? 'enabled' : '']">
            <i class="icon">directions_car</i>
          </div>
        </div>
      </div>
      <TimeInputField></TimeInputField>
      <div class="pure-u-1 pure-u-sm-3-24 time-option">
        <div>
          <input
            type="radio"
            id="search-forward"
            name="time-option"
            checked />
          <label for="search-forward">{{ $t.departure }}</label>
        </div>
        <div>
          <input type="radio" id="search-backward" name="time-option" />
          <label for="search-backward">{{ $t.arrival }}</label>
        </div>
      </div>
    </div>

    <div class="mode-picker-editor visible" v-show="isOptionsWindowOpened">
      <div class="header">
        <div class="sub-overlay-close">
          <i class="icon" @click="optionsWindowCloseClick">close</i>
        </div>
        <div class="title">
          {{ pressedOptions == firstOptions ? $t.startTransports : $t.destinationTransports }}
        </div>
      </div>
      <div class="content">
        <BlockWithCheckbox :title="$t.walk" :isChecked="pressedOptions.foot" @isCheckedChanged="pressedOptions.foot = $event">
          <div class="option">
            <div class="label">
              {{ $t.profile }}
            </div>
            <div class="profile-picker">
              <select>
                <option value="default">
                  {{ $t.searchProfile_default }}
                </option>
                <option value="accessibility1">
                  {{ $t.searchProfile_accessibility }}
                </option>
                <option value="wheelchair">
                  {{ $t.searchProfile_wheelchair }}
                </option>
                <option value="elevation">
                  {{ $t.searchProfile_elevation }}
                </option>
              </select>
            </div>
          </div>
          <Slider></Slider>
        </BlockWithCheckbox>
        <BlockWithCheckbox
          :title="$t.bike"
          :isChecked="pressedOptions.bicycle"
          @isCheckedChanged="pressedOptions.bicycle = $event">
          <Slider></Slider>
        </BlockWithCheckbox>
        <BlockWithCheckbox :title="$t.car" :isChecked="pressedOptions.car" @isCheckedChanged="pressedOptions.car = $event">
          <Slider></Slider>
          <div class="option">
            <label> <input type="checkbox" />{{ $t.useParking }} </label>
          </div>
        </BlockWithCheckbox>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import InputField from "../components/InputField.vue";
import BlockWithCheckbox from "../components/BlockWithCheckbox.vue";
import Slider from "../components/Slider.vue";
import Calendar from "../components/Calendar.vue";
import TimeInputField from "../components/TimeInputField.vue"

export default defineComponent({
  name: "ConnectionSearch",
  components: {
    InputField,
    BlockWithCheckbox,
    Slider,
    Calendar,
    TimeInputField,
  },
  data() {
    return {
      start: "",
      destination: "",
      isOptionsWindowOpened: false,
      pressedOptions: {
        foot: false,
        bicycle: false,
        car: false,
      } as OptionsButtons,
      firstOptions: {
        foot: true,
        bicycle: false,
        car: false,
      } as OptionsButtons,
      secondOptions: {
        foot: true,
        bicycle: false,
        car: false,
      } as OptionsButtons,
    };
  },
  methods: {
    swapStartDest() {
      let temp: string = this.start;
      this.start = this.destination;
      this.destination = temp;
    },
    setStartInput(input: string) {
      this.start = input;
    },
    setDestInput(input: string) {
      this.destination = input;
    },
    optinsButton1Click() {
      this.isOptionsWindowOpened = true;
      this.pressedOptions = this.firstOptions;
    },
    optinsButton2Click() {
      this.isOptionsWindowOpened = true;
      this.pressedOptions = this.secondOptions;
    },
    optionsWindowCloseClick() {
      this.isOptionsWindowOpened = false;
    },
  },
});


interface OptionsButtons {
  foot: boolean;
  bicycle: boolean;
  car: boolean;
}
</script>
