<template>
  <div :class="['overlay-container', isHidden ? 'hidden' : '']">
    <div class="overlay">
      <div id="overlay-content">
        <div id="search">
          <div class="pure-g-gutters">
            <div class="pure-u-1 pure-u-sm-12-24 from-location">
              <InputField labelName="Start" iconType="place" :showLabel="true" :initInputText="start" @inputChanged="setStartInput" />
              <div class="mode-picker-btn" @click="optinsButton1Click">
                <div :class="['mode', firstOptions.foot ? 'enabled' : '']"><i class="icon">directions_walk</i></div>
                <div :class="['mode', firstOptions.bicycle ? 'enabled' : '']"><i class="icon">directions_bike</i></div>
                <div :class="['mode', firstOptions.car ? 'enabled' : '']"><i class="icon">directions_car</i></div>
              </div>
            </div>
            <calendar class="pure-u-1 pure-u-sm-12-24"></calendar>
          </div>

          <button class="swap-locations-btn">
            <label class="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select">
              <input type="checkbox" @click="swapStartDest" />
              <i class="icon">swap_vert</i>
            </label>
          </button>

          <div class="pure-g-gutters">
            <div class="pure-u-1 pure-u-sm-12-24 from-location">
              <InputField labelName="Ziel" iconType="place" :showLabel="true" :initInputText="destination" @inputChanged="setDestInput" />
              <div class="mode-picker-btn" @click="optinsButton2Click">
                <div :class="['mode', secondOptions.foot ? 'enabled' : '']"><i class="icon">directions_walk</i></div>
                <div :class="['mode', secondOptions.bicycle ? 'enabled' : '']"><i class="icon">directions_bike</i></div>
                <div :class="['mode', secondOptions.car ? 'enabled' : '']"><i class="icon">directions_car</i></div>
              </div>
            </div>
            <InputField labelName="Uhrzeit" iconType="schedule" :showLabel="true" :initInputText="time"
              class="pure-u-1 pure-u-1 pure-u-sm-9-24"  />
          </div>

          <div class="mode-picker-editor visible" v-show="isOptionsWindowOpened">
            <div class="header">
              <div class="sub-overlay-close"><i class="icon" @click="optionsWindowCloseClick">close</i></div>
              <div class="title">Verkehrsmittel am Start</div>
            </div>
            <div class="content">
              <BlockWithCheckbox title="Fußweg" :isChecked="pressedOptions.foot">
                <div class="option">
                  <div class="label">Profil</div>
                  <div class="profile-picker">
                    <select>
                      <option value="default">Standard</option>
                      <option value="accessibility1">Auch nach leichten Wegen suchen</option>
                      <option value="wheelchair">Rollstuhl</option>
                      <option value="elevation">Weniger Steigung</option>
                    </select>
                  </div>
                </div>
                <Slider></Slider>
              </BlockWithCheckbox>
              <BlockWithCheckbox title="Fahrrad" :isChecked="pressedOptions.bicycle">
                <Slider></Slider>
              </BlockWithCheckbox>
              <BlockWithCheckbox title="Auto" :isChecked="pressedOptions.car">
                <Slider></Slider>
                <div class="option">
                  <label> <input type="checkbox" />Parkplätze verwenden </label>
                </div>
              </BlockWithCheckbox>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="overlay-tabs">
      <div class="overlay-toggle">
        <i class="icon" v-on:click="isHidden = !isHidden">arrow_drop_down</i>
      </div>
      <div class="trip-search-toggle">
        <i class="icon">train</i>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import InputField from "./InputField.vue";
import BlockWithCheckbox from "./BlockWithCheckbox.vue";
import Slider from "./Slider.vue";
import Calendar from './Calendar.vue'

export default defineComponent({
  name: "LeftMenu",
  components: {
    InputField,
    BlockWithCheckbox,
    Slider,
    Calendar
  },
  data() {
    return {
      start: '',
      destination: '',
      time:  (new Date).getHours() + ":" + ("0" + (new Date).getMinutes()).slice(-2),
      isOptionsWindowOpened: false,
      pressedOptions: {} as OptionsButtons,
      firstOptions: 
      {
        foot: true,
        bicycle: false,
        car: false
      } as OptionsButtons,
      secondOptions: 
      {
        foot: true,
        bicycle: false,
        car: false
      } as OptionsButtons,
      isHidden: false,
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
    }
  },
});

interface OptionsButtons {
  foot: Boolean,
  bicycle: Boolean,
  car: Boolean
}
</script>

<style>
</style>
