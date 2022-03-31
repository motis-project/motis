<template>
  <div>
    <InputField
      :showLabel="true"
      :labelName="$t.date"
      iconType="event"
      :isTimeCalendarField="true"
      :showArrows="true"
      :initInputText="dateToDisplay"
      @inputChangedNative="onFieldInput"
      @focus="calendarVisible = true"
      @blur="inputBluredHandler"
      @decreaseClick="stopInterval(-1)"
      @increaseClick="stopInterval(1)"
      @decreaseMouseDown="mouseDown(-1)"
      @increaseMouseDown="mouseDown(1)"
      @keydown="onKeyDown"
      @mouseup="onMouseUp"
      :showAutocomplete="false"></InputField>
    <div class="paper calendar" v-show="calendarVisible" @mousedown="calendarClickedHandler">
      <div class="month">
        <i class="icon" @click="changeMonth(-1)">chevron_left</i>
        <span class="month-name">{{ currentMonthToDisplay }}</span>
        <i class="icon" @click="changeMonth(1)">chevron_right</i>
      </div>
      <ul class="weekdays">
        <li v-for="dayName in weekDayNames" :key="dayName">
          {{ dayName }}
        </li>
      </ul>
      <ul class="calendardays">
        <li
          v-for="day in daysToDisplay"
          :key="day.valueOf()"
          :class="[
            day.getMonth() === currentMonth ? 'in-month' : 'out-of-month',
            compareDayWithInterval(day) ? 'valid-day' : 'invalid-day',
            day.getTime() == currentDate.getTime() ? 'selected' : '',
          ]"
          @mousedown.stop="dayClick(day)">
          {{ day.getDate() }}
        </li>
      </ul>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import InputField from "./InputField.vue";

export default defineComponent({
  name: "Calendar",
  components: {
    InputField,
  },
  emits: ["dateChanged"],
  data() {
    return {
      daysToDisplay: [] as Date[],
      currentMonthToDisplay: "",
      weekDayNames: [] as string[],
      currentMonth: 0,
      calendarVisible: false,
      currentDate: {} as Date,
      calendarClicked: false,
      timeout: 0,
      interval: 0,
      beginIntervalDate: 0 as number,
      endIntervalDate: 0 as number,
      prevString: "..",
      dateToDisplay: ""
    };
  },
  watch: {
    currentDate: function (date: Date) {
      this.dateToDisplay = this.$ds.getDateString(this.currentDate.valueOf());
      this.prevString = this.dateToDisplay;
      this.daysToDisplay = [];
      this.weekDayNames = [];
      this.currentMonthToDisplay = `${date.toLocaleString(this.$ts.currentLocale, { month: "long" })} ${date.getFullYear()}`;
      this.currentMonth = date.getMonth();
      let day = new Date(date.getFullYear(), date.getMonth(), 1,
                         this.currentDate.getHours(), this.currentDate.getMinutes(), this.currentDate.getSeconds(), this.currentDate.getMilliseconds());
      let first = day.getDate() - (day.getDay() === 0 ? 7 : day.getDay());
      day = new Date(day.setDate(first));
      for (let i = 0; i < 42; i++) {
        this.daysToDisplay.push(day);
        day = new Date(day.setDate(day.getDate() + 1));
        if((i === 35 || i === 28) && day.getMonth() !== this.currentMonth) {
          this.daysToDisplay.pop();
          break;
        }
        if (i < 7) {
          this.weekDayNames.push(day.toLocaleString(this.$ts.currentLocale, { weekday: "short" }));
        }
      }
    },
  },
  created() {
    this.currentDate = this.$ds.date;
    this.beginIntervalDate = this.$ds.intervalFromServer.begin * 1000;
    this.endIntervalDate = this.$ds.intervalFromServer.end * 1000;
    this.prevString = this.$ds.getDateString(this.currentDate.valueOf());
    this.dateToDisplay = this.$ds.getDateString(this.currentDate.valueOf())
  },
  methods: {
    changeMonth(change: number) {
      this.currentDate = new Date(
        this.currentDate.getFullYear(),
        this.currentDate.getMonth() + change,
        this.currentDate.getDate()
      );
      this.$emit('dateChanged', this.currentDate);
    },
    changeDay(change: number) {
      this.currentDate = new Date(
        this.currentDate.getFullYear(),
        this.currentDate.getMonth(),
        this.currentDate.getDate() + change
      );
      this.$emit('dateChanged', this.currentDate);
    },
    calendarClickedHandler() {
      this.calendarClicked = true;
    },
    inputBluredHandler(event: Event) {
      if (this.calendarClicked) {
        (<HTMLElement>event.target).focus();
        this.calendarClicked = false;
      }
      else {
        this.calendarVisible = false;
        this.setDateFromString(this.prevString, true);
      }
    },
    onFieldInput(event: Event) {
      let inputField = event.target as HTMLInputElement;
      let value = inputField.value;
      if (value === ""){
        value = "..";
        this.setDateToDisplay(value, inputField, 0);
        this.prevString = value;
        return;
      }
      if(value === this.prevString) {
        return;
      }
      if(!this.setDateFromString(value)) {
        let arr = value.split(".");
        if(arr.length !== 3) {
          value = this.prevString;
          this.setDateToDisplay(this.prevString);
        }
        else {
          let reg = /^\d+$/ ;
          let arrInt = arr.map<(number | undefined)>(s => Number.parseInt(s));
          if((arr[0].length !== 0 && !arr[0].match(reg)) || arrInt[0] === undefined || arrInt[0] < 0 || arrInt[0] > 31
            || (arr[1].length !== 0 && !arr[1].match(reg)) || arrInt[1] === undefined || arrInt[1] < 0 || arrInt[1] > 12
            || (arr[2].length !== 0 && !arr[2].match(reg)) || arrInt[2] === undefined || arrInt[2] < 0
            || arr[0].length > 2 || arr[1].length > 2 || arr[2].length > 4) {
            value = this.prevString;
            this.setDateToDisplay(this.prevString);
          }
          arr = value.split(".");
          arrInt = arr.map<(number | undefined)>(s => Number.parseInt(s));
          if(arrInt[0] !== undefined && arrInt[0] > 3 && arrInt[0] < 10 && arr[0].length !== 2){
            arr[0] = "0" + arrInt[0];
            value = arr[0] + "." + arr[1] + "." + arr[2];
            this.setDateToDisplay(value, inputField, 3);
          }
          if(arrInt[1] !== undefined && arrInt[1] > 1 && arrInt[1] < 10 && arr[1].length !== 2){
            arr[1] = "0" + arrInt[1];
            value = arr[0] + "." + arr[1] + "." + arr[2];
            this.setDateToDisplay(value, inputField, 6);
          }
          if(arrInt[2] !== undefined && arr[2].length === 2 && arrInt[2] !== 20){
            arr[2] = "20" + arrInt[2];
            value = arr[0] + "." + arr[1] + "." + arr[2];
            this.setDateToDisplay(value);
          }
        }
      }
      this.prevString = value;
    },
    onKeyDown(event: KeyboardEvent){
      let inputField = event.target as HTMLInputElement;
      let value = inputField.value;
      let arr = value.split(".");
      let lastValidDate = this.$ds.getDateString(this.currentDate.valueOf());
      let arrLastValidDate = lastValidDate.split(".");
      if(event.key === "Backspace"){
        if(inputField.selectionStart === 3){
          inputField.setSelectionRange(2, 2);
        }
        if(inputField.selectionStart === 6){
          inputField.setSelectionRange(5, 5);
        }
      }
      if(!Number.isNaN(Number.parseInt(event.key)) && inputField.selectionStart === 2) {
        inputField.setSelectionRange(3, 3);
      }
      if(!Number.isNaN(Number.parseInt(event.key)) && inputField.selectionStart === 5) {
        inputField.setSelectionRange(6, 6);
      }
      if(event.key ==="ArrowRight"){
        if(Number.parseInt(arr[0]) > 0 && Number.parseInt(arr[0]) <=3 && inputField.selectionStart === 1 && arr[0].length < 2){
          arr[0] = "0" + arr[0];
          value = arr[0] + "." + arr[1] + "." + arr[2];
          this.setDateToDisplay(value, inputField, 2);
        }
        if(Number.parseInt(arr[1]) === 1 && inputField.selectionStart === 4 && arr[1].length < 2){
          arr[1] = "0" + arr[1];
          value = arr[0] + "." + arr[1] + "." + arr[2];
          this.setDateToDisplay(value);
        }
        if(arr[0] === "" && inputField.selectionStart === 0){
          value = arrLastValidDate[0] + "." + arr[1] + "." + arr[2];
          this.setDateToDisplay(value, inputField, 2);
        }
        if(arr[0] === "0" && inputField.selectionStart === 1){
          value = arrLastValidDate[0] + "." + arr[1] + "." + arr[2];
          this.setDateToDisplay(value, inputField, 2);
        }
        if(arr[1] === "" && inputField.selectionStart === 3){
          value = arr[0] + "." + arrLastValidDate[1] + "." + arr[2];
          this.setDateToDisplay(value, inputField, 5);
        }
        if(arr[1] === "0" && inputField.selectionStart === 4){
          value = arr[0] + "." + arrLastValidDate[1] + "." + arr[2];
          this.setDateToDisplay(value, inputField, 5);
        }
      }
      if(event.key ==="ArrowLeft"){
        if(Number.parseInt(arr[2]) >= 0 && inputField.selectionStart === 6){
          if (arr[2].length === 2){
            arr[2] = "20" + arr[2];
            value = arr[0] + "." + arr[1] + "." + arr[2];
            this.setDateToDisplay(value, inputField, 6);
          }
          if (arr[2].length === 1){
            arr[2] = "200" + arr[2];
            value = arr[0] + "." + arr[1] + "." + arr[2];
            this.setDateToDisplay(value, inputField, 6);
          }
        }
        if (arr[1] === "0" && inputField.selectionStart === 3) {
          value = arr[0] + "." + arrLastValidDate[1] + "." + arr[2];
          this.setDateToDisplay(value, inputField, 3);
        }
        if (arr[1] === "1" && inputField.selectionStart === 3) {
          arr[1] = "0" + arr[1];
          value = arr[0] + "." + arr[1] + "." + arr[2];
          this.setDateToDisplay(value, inputField, 3);
        }
      }
      this.prevString = value;
    },
    onMouseUp(event: MouseEvent) {
      if(event.button === 0){
        let inputField = event.target as HTMLInputElement;
        let value = inputField.value;
        let arr = value.split(".");
        let lastValidDate = this.$ds.getDateString(this.currentDate.valueOf());
        let arrLastValidDate = lastValidDate.split(".");
        let position = inputField.selectionStart !== null ? inputField.selectionStart : 0;
        if(Number.parseInt(arr[0]) > 0 && Number.parseInt(arr[0]) <=3 && inputField.selectionStart !== 1 && inputField.selectionStart !== 0 && arr[0].length === 1){
          arr[0] = "0" + arr[0];
          value = arr[0] + "." + arr[1] + "." + arr[2];
          this.setDateToDisplay(value, inputField, position + 1);
        }
        if(arr[0] === "" && inputField.selectionStart !== 0){
          value = arrLastValidDate[0] + "." + arr[1] + "." + arr[2];
          this.setDateToDisplay(value, inputField, position + 2);
        }
        if(arr[0] === "0" && inputField.selectionStart !== 0 && inputField.selectionStart !== 1){
          value = arrLastValidDate[0] + "." + arr[1] + "." + arr[2];
          this.setDateToDisplay(value, inputField, position + 1);
        }
        if(Number.parseInt(arr[1]) === 1 && inputField.selectionStart !== 3 && inputField.selectionStart !== 4 && arr[1].length === 1) {
          arr[1] = "0" + arr[1];
          value = arr[0] + "." + arr[1] + "." + arr[2];
          if(position < 3){
            this.setDateToDisplay(value, inputField, position);
          }
          else{
            this.setDateToDisplay(value, inputField, position + 1);
          }
        }
        if(arr[1] === "" && inputField.selectionStart !== 3){
          value = arr[0] + "." + arrLastValidDate[1] + "." + arr[2];
          if(position < 3){
            this.setDateToDisplay(value, inputField, position);
          }
          else{
            this.setDateToDisplay(value, inputField, position + 2);
          }
        }
        if(arr[1] === "0" && inputField.selectionStart !== 3 && inputField.selectionStart !== 4){
          value = arr[0] + "." + arrLastValidDate[1] + "." + arr[2];
          if(position < 3){
            this.setDateToDisplay(value, inputField, position);
          }
          else{
            this.setDateToDisplay(value, inputField, position + 1);
          }
        }
        if (arr[2] === "20" && inputField.selectionStart !== 6 && inputField.selectionStart !== 7 && inputField.selectionStart !== 8){
          value = arr[0] + "." + arr[1] + ".2020";
          this.setDateToDisplay(value, inputField, position);
        }
        if (arr[2].length === 1 && inputField.selectionStart !== 6 && inputField.selectionStart !== 7){
          arr[2] = "200" + arr[2];
          value = arr[0] + "." + arr[1] + "." + arr[2];
          this.setDateToDisplay(value, inputField, position);
        }

      }
    },
    dayClick(day : Date) {
      this.currentDate = day;
      this.calendarVisible = false;
    },
    mouseDown(value: number){
      this.timeout = setTimeout(() => this.interval = setInterval(() => this.changeDay(value), 100), 1000)
    },
    stopInterval(value: number){
      clearInterval(this.interval);
      clearTimeout(this.timeout)
      this.changeDay(value);
    },
    compareDayWithInterval(date: Date) : boolean {
      let currentDay = date.valueOf();
      if(currentDay >= this.beginIntervalDate && currentDay < this.endIntervalDate){
        return true;
      }
      return false;
    },
    setDateToDisplay(value: string, inputField?: HTMLInputElement, cursorPosition?: number) {
      this.dateToDisplay = "";
      this.$nextTick(() => {
        this.dateToDisplay = value;
        this.setDateFromString(value);
        if(inputField && cursorPosition !== undefined) {
          this.$nextTick(() => inputField.setSelectionRange(cursorPosition, cursorPosition));
        }
      });
    },
    setDateFromString(value: string, setDateStringOthervise?: boolean): boolean {
      let d = this.$ds.parseDate(value);
      if(d.valueOf()) {
        this.currentDate = d;
        this.$emit("dateChanged", this.currentDate);
      }
      else if(setDateStringOthervise) {
        this.setDateToDisplay(this.$ds.getDateString(this.currentDate.valueOf()));
      }
      return !!d.valueOf();
    }
  },
});
</script>

<style>
</style>
