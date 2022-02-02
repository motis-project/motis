<template>
  <InputField
    :labelName="$t.time"
    iconType="schedule"
    :showLabel="true"
    :initInputText="timeToDisplay"
    :isTimeCalendarField="true"
    :showArrows="true"
    @inputChangedNative="setTime"
    @decreaseClick="changeTime(-1)"
    @increaseClick="changeTime(1)"
    :showAutocomplete="false"
    :key="inputFieldKey"
    @blur="inputFieldKey++"
    @keydown="onKeyDown"
    @mouseup="onMouseUp"></InputField>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import InputField from "./InputField.vue";

export default defineComponent({
  name: "TimeInputField",
  components: {
    InputField,
  },
  emits: [
    "timeChanged"
  ],
  data() {
    return {
      time: {} as Date,
      inputFieldKey: 0,
      prevString: ":",
    };
  },
  computed: {
    timeToDisplay: function (): string {
      let result = this.$ds.getTimeString(this.time.valueOf());
      return result;
    },
  },
  created() {
    let currentTime = this.$ds.date;
    this.time = currentTime;
    this.prevString = this.timeToDisplay;
  },
  methods: {
    changeTime(change: number) {
      this.time = new Date(
        this.time.getFullYear(),
        this.time.getMonth(),
        this.time.getDay(),
        this.time.getHours() + change,
        this.time.getMinutes()
      );
      this.$emit("timeChanged", this.time);
    },
    setTime(event: Event) {
      let inputField = event.target as HTMLInputElement;
      let value = inputField.value
      if(value === this.prevString) {
        return;
      }
      let t = this.$ds.parseTime(value);

      if(t.valueOf()) {
        this.time = t;
        this.$emit("timeChanged", this.time);
      }
      else {
        let arr = value.split(":");
        if(arr.length !== 2) {
          value = this.prevString;
          inputField.value = this.prevString;
        }
        else {
          let arrInt = arr.map<(number | undefined)>(s => Number.parseInt(s));
          let reg = /^\d+$/
          if((arr[0].length !== 0 && !arr[0].match(reg)) || arrInt[0] === undefined || arrInt[0] < 0 || arrInt[0] >= 24
            || (arr[1].length !== 0 && !arr[1].match(reg)) || arrInt[1] === undefined || arrInt[1] < 0 || arrInt[1] >= 60
            || arr[0].length > 2 || arr[1].length > 2) {
            value = this.prevString;
            inputField.value = this.prevString;
          }
          arr = value.split(":");
          arrInt = arr.map<(number | undefined)>(s => Number.parseInt(s));
          if(arrInt[0] !== undefined && arrInt[0] > 2 && arrInt[0] < 10){
            arr[0] = "0" + arrInt[0];
            value = arr[0] + ":" + arr[1];
            inputField.value = value;
          }
          if(arrInt[1] !== undefined && arrInt[1] > 5 && arrInt[1] < 10){
            arr[1] = "0" + arrInt[1];
            value = arr[0] + ":" + arr[1];
            inputField.value = value;
          }
        }

      }
      this.prevString = value;

    },
    onKeyDown(event: KeyboardEvent){
      let inputField = event.target as HTMLInputElement;
      let value = inputField.value;
      let arr = value.split(":");
      let lastValidTime = this.$ds.getTimeString(this.time.valueOf());
      let arrLastValidTime = lastValidTime.split(":");
      if(event.key === "Backspace"){
        if(inputField.selectionStart === 3){
          inputField.setSelectionRange(2, 2);
        }
      }
      if(!Number.isNaN(Number.parseInt(event.key)) && inputField.selectionStart === 2) {
        inputField.setSelectionRange(3, 3);
      }
      if(event.key ==="ArrowRight"){
        if(Number.parseInt(arr[0]) >= 0 && Number.parseInt(arr[0]) <=2 && inputField.selectionStart === 1){
          arr[0] = "0" + arr[0];
          value = arr[0] + ":" + arr[1];
          inputField.value = value;
        }
        if(arr[0] === "" && inputField.selectionStart === 0){
          value = arrLastValidTime[0] + ":" + arr[1];
          inputField.value = value;
          inputField.setSelectionRange(2, 2);
        }
      }
      if(event.key ==="ArrowLeft" && Number.parseInt(arr[1]) >= 0 && Number.parseInt(arr[1]) <=5
        && inputField.selectionStart === 3){
        arr[1] = "0" + arr[1];
        value = arr[0] + ":" + arr[1];
        inputField.value = value;
        inputField.setSelectionRange(3, 3);
      }
      if((event.key === "4" || event.key === "5" || event.key === "6" || event.key === "7" || event.key === "8" || event.key === "9")
        && arr[0] === "2" && inputField.selectionStart === 1){
        arr[0] = "02";
        value = arr[0] + ":" + arr[1];
        inputField.value = value;
      }
      this.prevString = value;
    },
    onMouseUp(event: MouseEvent) {
      if(event.button === 0){
        let inputField = event.target as HTMLInputElement;
        let value = inputField.value;
        let arr = value.split(":");
        let lastValidTime = this.$ds.getTimeString(this.time.valueOf());
        let arrLastValidTime = lastValidTime.split(":");
        let position = inputField.selectionStart !== null ? inputField.selectionStart : 0;
        if(Number.parseInt(arr[0]) >= 0 && Number.parseInt(arr[0]) <=2 && inputField.selectionStart !== 1 && inputField.selectionStart !== 0){
          if(arr[0].length === 1){
            arr[0] = "0" + arr[0];
            value = arr[0] + ":" + arr[1];
            inputField.value = value;
            inputField.setSelectionRange(position + 1, position + 1);
          }
        }
        if(arr[0] === "" && inputField.selectionStart !== 0){
          value = arrLastValidTime[0] + ":" + arr[1];
          inputField.value = value;
          inputField.setSelectionRange(position + 2, position + 2);
        }
        if(Number.parseInt(arr[1]) >= 0 && Number.parseInt(arr[1]) <=5 && inputField.selectionStart !== 3
          && inputField.selectionStart !== 4 && arr[1].length === 1) {
          arr[1] = "0" + arr[1];
          value = arr[0] + ":" + arr[1];
          inputField.value = value;
          inputField.setSelectionRange(position, position);
        }
      }
    }
  },
});
</script>
