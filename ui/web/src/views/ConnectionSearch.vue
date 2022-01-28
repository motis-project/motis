<template>
  <div id="search">
    <div class="main-gutter from-gutter">
      <InputField
        :labelName="$t.start"
        iconType="place"
        :isTimeCalendarField="false"
        :showLabel="true"
        :initInputText="start"
        @inputChanged="setStartInput"
        :showAutocomplete="true"
        @autocompleteElementClicked="startObjectClicked"></InputField>

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
    <Calendar class="main-gutter calendar-gutter" @dateChanged="dateChanged"></Calendar>

    <div class="main-gutter to-gutter">
      <InputField
        :labelName="$t.destination"
        iconType="place"
        :showLabel="true"
        :initInputText="destination"
        @inputChanged="setDestInput"
        :showAutocomplete="true"
        @autocompleteElementClicked="endObjectClicked"></InputField>
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
    <TimeInputField @timeChanged="timeChanged" class="main-gutter time-gutter"></TimeInputField>
    <div class="main-gutter time-options-gutter">
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

    <div class="mode-picker-editor visible" v-show="isOptionsWindowOpened">
      <div class="header">
        <div class="sub-overlay-close">
          <i class="icon" @click="optionsWindowCloseClick">close</i>
        </div>
        <div class="title">
          {{
            pressedOptions == firstOptions
              ? $t.startTransports
              : $t.destinationTransports
          }}
        </div>
      </div>
      <div class="content">
        <BlockWithCheckbox :title="$t.walk" :isChecked="pressedOptions.foot" @isCheckedChanged="pressedOptions.foot = $event">
          <div class="option">
            <div class="label">
              {{ $t.profile }}
            </div>
            <div class="profile-picker">
              <select v-model="pressedOptions.footProfile">
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
          <Slider
            :initSliderValue="pressedOptions.footDuration"
            @sliderValueChanged="pressedOptions.footDuration = $event"></Slider>
        </BlockWithCheckbox>
        <BlockWithCheckbox
          :title="$t.bike"
          :isChecked="pressedOptions.bicycle"
          @isCheckedChanged="pressedOptions.bicycle = $event">
          <Slider
            :initSliderValue="pressedOptions.bicycleDuration"
            @sliderValueChanged="pressedOptions.bicycleDuration = $event"></Slider>
        </BlockWithCheckbox>
        <BlockWithCheckbox
          :title="$t.car"
          :isChecked="pressedOptions.car"
          @isCheckedChanged="pressedOptions.car = $event">
          <Slider
            :initSliderValue="pressedOptions.carDuration"
            @sliderValueChanged="pressedOptions.carDuration = $event"></Slider>
          <div class="option">
            <label> <input type="checkbox" v-model="pressedOptions.carParking" />{{ $t.useParking }} </label>
          </div>
        </BlockWithCheckbox>
      </div>
    </div>
  </div>
  <LoadingBar v-if="contentLoadingState === LoadingState.Loading"></LoadingBar>
  <div v-else-if="contentLoadingState === LoadingState.Loaded" id="connections">
    <div class="connections">
      <div class="extend-search-interval search-before">
        <a>{{ $t.earlier }}</a>
      </div>
      <div class="connection-list">
        <div class="date-header divider">
          <span>{{ $ds.getDateString(connections[0].stops[0].departure.time * 1000) }}</span>
        </div>
        <div
          class="connection"
          v-for="(c, cIndex) in connections"
          :key="c"
          @click="connectionClicked(cIndex)">
          <div class="pure-g">
            <div class="pure-u-4-24 connection-times">
              <div class="connection-departure">
                {{ $ds.getTimeString(c.stops[0].departure.time * 1000) }}
              </div>
              <div class="connection-arrival">
                {{ $ds.getTimeString(c.stops[c.stops.length - 1].arrival.time * 1000) }}
              </div>
            </div>
            <div class="pure-u-4-24 connection-duration">
              <div>{{ getReadableDuration(c.stops[0].departure.time, c.stops[c.stops.length - 1].arrival.time, $ts) }}</div>
            </div>
            <div class="pure-u-16-24 connection-trains">
              <div class="transport-graph">
                <svg width="335" height="40" viewBox="0 0 335 40">
                  <g>
                    <TransportLine
                      v-for="(t, tIndex) in getNonEmptyTransports(c.transports)"
                      :key="t.range"
                      :move="t"
                      :allStops="c.stops"
                      :lineIndex="tIndex"
                      :lineCount="getNonEmptyTransports(c.transports).length"
                      @mouseEnter="showTooltip($event, cIndex)"
                      @mouseLeave="isTooltipVisible[cIndex] = false"></TransportLine>
                  </g>
                  <g class="destination">
                    <circle cx="329" cy="12" r="6"></circle>
                  </g>
                </svg>
                <div
                  :class="['tooltip', isTooltipVisible[cIndex] ? 'visible' : '']"
                  :style="`position: absolute; left: ${transportTooltipInfo.x}px; top: 23px`">
                  <div class="stations">
                    <div class="departure">
                      <span class="station">{{ transportTooltipInfo.start }}</span><span class="time">{{ transportTooltipInfo.departure }}</span>
                    </div>
                    <div class="arrival">
                      <span class="station">{{ transportTooltipInfo.destination }}</span><span class="time">{{ transportTooltipInfo.arrival }}</span>
                    </div>
                  </div>
                  <div class="transport-name">
                    <span>{{ transportTooltipInfo.transportName }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="divider footer"></div>
      <div class="extend-search-interval search-after">
        <a>{{ $t.later }}</a>
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
import TimeInputField from "../components/TimeInputField.vue";
import StationGuess from "../models/StationGuess";
import AddressGuess from "../models/AddressGuess";
import { Start, Mode, InputStation } from "../models/ConnectionContent";
import Position from "../models/SmallTypes/Position";
import TripResponseContent, { Move } from "../models/TripResponseContent";
import WayMixin from "../mixins/WayMixin";
import TransportLine from "../components/TransportLine.vue";
import LoadingBar, { LoadingState } from "../components/LoadingBar.vue"
import Transport from "../models/Transport";
import CustomMovement from "../models/CustomMovement";

export default defineComponent({
  name: "ConnectionSearch",
  components: {
    InputField,
    BlockWithCheckbox,
    Slider,
    Calendar,
    TimeInputField,
    TransportLine,
    LoadingBar
  },
  mixins: [ WayMixin ],
  data() {
    return {
      start: "",
      destination: "",
      startObject: {} as StationGuess | AddressGuess,
      destinationObject: {} as StationGuess | AddressGuess,
      isOptionsWindowOpened: false,
      pressedOptions: {
        foot: true,
        bicycle: false,
        car: false,
        footDuration: 15,
        bicycleDuration: 15,
        carDuration: 15,
        footProfile: "default",
        carParking: true,
      } as OptionsButtons,
      firstOptions: {
        foot: true,
        bicycle: false,
        car: false,
        footDuration: 15,
        bicycleDuration: 15,
        carDuration: 15,
        footProfile: "default",
        carParking: true,
      } as OptionsButtons,
      secondOptions: {
        foot: true,
        bicycle: false,
        car: false,
        footDuration: 15,
        bicycleDuration: 15,
        carDuration: 15,
        footProfile: "default",
        carParking: true,
      } as OptionsButtons,
      dateTime: this.$ds.date,
      timeoutIndex: -1,
      connections: [] as TripResponseContent[],
      contentLoadingState: LoadingState.NothingToShow,
      LoadingState: LoadingState,
      isTooltipVisible: [] as boolean[],
      transportTooltipInfo: {} as TransportTooltipInfo
    };
  },
  activated() {
    for(let i = 0; i < this.isTooltipVisible.length; i++) {
      this.isTooltipVisible[i] = false;
    }
  },
  created() {
    window.addEventListener("dragover", (event: Event) => event.preventDefault());
    window.addEventListener("dragenter", (event: Event) => event.preventDefault());
    window.addEventListener("drop", this.onDrop);
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
      this.sendRequest();
    },
    dateChanged(date: Date) {
      this.dateTime = new Date(
        date.getFullYear(),
        date.getMonth(),
        date.getDate(),
        this.dateTime.getHours(),
        this.dateTime.getMinutes(),
        this.dateTime.getSeconds(),
        this.dateTime.getMilliseconds());
      this.sendRequest();
    },
    timeChanged(time: Date) {
      this.dateTime = new Date(
        this.dateTime.getFullYear(),
        this.dateTime.getMonth(),
        this.dateTime.getDate(),
        time.getHours(),
        time.getMinutes(),
        this.dateTime.getSeconds(),
        this.dateTime.getMilliseconds());
      this.sendRequest();
    },
    startObjectClicked(startObject: StationGuess | AddressGuess) {
      this.startObject = startObject;
      this.$store.state.startInput = startObject;
      this.sendRequest();
    },
    endObjectClicked(destinationObject: StationGuess | AddressGuess) {
      this.destinationObject = destinationObject;
      this.$store.state.destinationInput = destinationObject;
      this.sendRequest();
    },
    sendRequest() {
      if(this.start !== "" && this.destination !== "") {
        this.contentLoadingState = LoadingState.Loading;
        this.isTooltipVisible = []
        if(this.timeoutIndex !== -1) {
          clearTimeout(this.timeoutIndex);
        }
        let start = {
          interval: {
            begin: Math.floor(this.dateTime.valueOf() / 1000),
            end: Math.floor(this.dateTime.valueOf() / 1000) + 7200
          },
          /* eslint-disable camelcase*/
          min_connection_count: 5,
          extend_interval_earlier: true,
          extend_interval_later: true
        } as Start;
        if("id" in this.startObject) {
          start = {
            ...start,
            station: {
              name: this.startObject.name,
              id: this.startObject.id
            }
          }
        }
        else {
          start = {
            ...start,
            position: this.startObject.pos
          }
        }
        let destination: Position | InputStation;
        if("id" in this.destinationObject) {
          destination = {
            name: this.destinationObject.name,
            id: this.destinationObject.id
          }
        }
        else {
          destination = this.destinationObject.pos;
        }
        this.timeoutIndex = setTimeout(() => {
          this.timeoutIndex = -1;
          this.$postService.getConnectionResponse({
            start_type: "id" in this.startObject ? "PretripStart" : "IntermodalPretripStart",
            start: start,
            start_modes: this.getModesArray(this.firstOptions),
            destination_type: "id" in this.destinationObject ? "InputStation" : "InputPosition",
            destination: destination,
            destination_modes: this.getModesArray(this.secondOptions)
          }).then((data) => {
            this.setConnections(data.connections)
          })
        }, 500);
      }
    },
    setConnections(connections: TripResponseContent[]) {
      this.connections = connections;
      this.contentLoadingState = LoadingState.Loaded;
      for(let i = 0; i < this.connections.length; i++) {
        this.isTooltipVisible.push(false);
      }
      this.$store.state.connections = this.connections;
    },
    getModesArray(options: OptionsButtons) {
      let res: Mode[] = [];
      if(options.foot) {
        res.push({
          mode_type: "FootPPR",
          mode: {
            search_options: {
              profile: options.footProfile,
              duration_limit: options.footDuration * 60
            }
          }
        })
      }
      if(options.bicycle) {
        res.push({
          mode_type: "Bike",
          mode: {
            max_duration: options.bicycleDuration * 60
          }
        })
      }
      if(options.car) {
        if(!options.carParking) {
          res.push({
            mode_type: "Car",
            mode: {
              max_duration: options.carDuration * 60
            }
          })
        }
        else {
          res.push({
            mode_type: "CarParking",
            mode: {
              max_car_duration: options.carDuration * 60,
              ppr_search_options: {
                profile: "default",
                duration_limit: 300
              }
            }
          })
        }
      }
      return res;
      /* eslint-enable camelcase*/
    },
    showTooltip(event: LineMouseOverEvent, index: number) {
      this.isTooltipVisible[index] = true;
      const t = event.transport;
      const stops = this.connections[index].stops.slice(t.range.from, t.range.to + 1);
      if("clasz" in t) {
        this.transportTooltipInfo = {
          start: stops[0].station.name,
          destination: stops[stops.length - 1].station.name,
          departure: this.$ds.getTimeString(stops[0].departure.time * 1000),
          arrival: this.$ds.getTimeString(stops[stops.length - 1].arrival.time * 1000),
          transportName: t.name,
          x: event.x
        }
      }
      else {
        const transports = this.connections[index].transports.map(tr => tr.move);
        let start = stops[0].station.name;
        if(transports.indexOf(t) === 0) {
          start = this.startObject.name
        }
        else if(transports.indexOf(t) === 1 && "mumo_type" in transports[0] && transports[0].mumo_type === "car"
          || transports.indexOf(t) === transports.length - 1 && "mumo_type" in t && t.mumo_type === "car") {
          start = this.$t.parking
        }

        let destination = stops[stops.length - 1].station.name;
        let lastT = transports[transports.length - 1];

        if(transports.indexOf(t) === transports.length - 1) {
          destination = this.destinationObject.name;
        }
        else if(transports.indexOf(t) === transports.length - 2 && "mumo_type" in lastT && lastT.mumo_type === "car"
          || transports.indexOf(t) === 0 && "mumo_type" in t && t.mumo_type === "car") {
          destination = this.$t.parking
        }

        this.transportTooltipInfo = {
          start: start,
          destination: destination,
          departure: this.$ds.getTimeString(stops[0].departure.time * 1000),
          arrival: this.$ds.getTimeString(stops[stops.length - 1].arrival.time * 1000),
          transportName: t.mumo_type === "foot" ? this.$t.walk : this.$t[t.mumo_type],
          x: event.x
        }
      }
    },
    connectionClicked(connectionIndex: number) {
      this.$router.push({
        name: "Connection",
        params: {
          index: connectionIndex
        },
      })
    },
    getNonEmptyTransports(transports: Move[]): Move[] {
      const res: Move[] = [];
      for(let i = 0; i < transports.length; i++) {
        const t = transports[i];
        if(!("mumo_id" in t.move) || t.move.mumo_id !== -1) {
          res.push({...t})
        }
        else {
          res[res.length - 1].move.range.to = t.move.range.to;
        }
      }
      return res;
    },
    onDrop(event: DragEvent) {
      if(event.dataTransfer !== null && event.dataTransfer.files.length > 0) {
        event.preventDefault();
        console.log("Drop");
        event.dataTransfer.files[0].text().then(t => this.setConnections(
          (JSON.parse(t) as {
            content: {
              connections: TripResponseContent[]
            }
          }).content.connections))
      }
    }
  },
});

interface OptionsButtons {
  foot: boolean,
  bicycle: boolean,
  car: boolean,
  footDuration: number,
  bicycleDuration: number,
  carDuration: number,
  carParking: boolean,
  footProfile: "default" | "accessibility1" | "wheelchair" | "elevation"
}

interface LineMouseOverEvent {
  x: number,
  transport: Transport | CustomMovement
}

interface TransportTooltipInfo {
  start: string,
  destination: string,
  departure: string,
  arrival: string,
  transportName: string,
  x: number
}
</script>
