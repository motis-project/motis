<template>
  <div>
    <input-field
      :showLabel="true"
      labelName="Datum"
      iconType="event"
      :showArrows="true"
      :initInputText="currentDate.toLocaleString('de-DE', { month: '2-digit', year: 'numeric', day: 'numeric' })"
      @inputChanged="onFieldInput"
      @focus="calendarVisible = true"
      @blur="inputBluredHandler"
      @decreaseClick="changeDay(-1)"
      @increaseClick="changeDay(1)"
    ></input-field>
    <div class="paper calendar" v-show="calendarVisible" @mousedown="calendarClickedHandler">
      <div class="month">
        <i class="icon" @click="changeMonth(-1)">chevron_left</i>
        <span class="month-name">{{ currentMonthToDisplay }}</span>
        <i class="icon" @click="changeMonth(1)">chevron_right</i>
      </div>
      <ul class="weekdays">
        <li v-for="dayName in weekDayNames" :key="dayName">{{ dayName }}</li>
      </ul>
      <ul class="calendardays">
        <li
          v-for="day in daysToDisplay"
          :key="day.valueOf()"
          :class="[
            day.getMonth() === currentMonth ? 'in-month' : 'out-of-month',
            'valid-day',
            day.getTime() == currentDate.getTime() ? 'selected' : '',
          ]"
          @mousedown.stop="dayClick(day)"
        >
          {{ day.getDate() }}
        </li>
      </ul>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, PropType } from "vue";
import InputField from "./InputField.vue";

export default defineComponent({
  name: "Calendar",
  components: {
    InputField,
  },
  data() {
    return {
      daysToDisplay: [] as Date[],
      currentMonthToDisplay: "",
      weekDayNames: [] as string[],
      currentMonth: 0,
      calendarVisible: false,
      currentDate: {} as Date,
      calendarClicked: false,
    };
  },
  created() {
    let now = new Date();
    this.currentDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  },
  watch: {
    currentDate: function (date: Date) {
      this.daysToDisplay = [];
      this.weekDayNames = [];
      this.currentMonthToDisplay = `${date.toLocaleString("de-DE", { month: "long" })} ${date.getFullYear()}`;
      this.currentMonth = date.getMonth();
      let day = new Date(date.getFullYear(), date.getMonth(), 1);
      let first = day.getDate() - (day.getDay() == 0 ? 7 : day.getDay());
      day = new Date(day.setDate(first));
      for (let i = 0; i < 42; i++) {
        this.daysToDisplay.push(day);
        day = new Date(day.setDate(day.getDate() + 1));
        if (i < 7) {
          this.weekDayNames.push(day.toLocaleString("de-DE", { weekday: "short" }));
        }
      }
    },
  },
  methods: {
    changeMonth(change: number) {
      this.currentDate = new Date(
        this.currentDate.getFullYear(),
        this.currentDate.getMonth() + change,
        this.currentDate.getDate()
      );
    },
    changeDay(change: number) {
      this.currentDate = new Date(
        this.currentDate.getFullYear(),
        this.currentDate.getMonth(),
        this.currentDate.getDate() + change
      );
    },
    calendarClickedHandler() {
      this.calendarClicked = true;
    },
    inputBluredHandler(event: Event) {
      if (this.calendarClicked) {
        (<HTMLElement>event.target).focus();
        this.calendarClicked = false;
      } else {
        this.calendarVisible = false;
      }
    },
    onFieldInput(value: string) {
      let arr = value.split(".");
      if (arr.length === 3 && arr[2].length === 4 && arr[1].length == 2) {
        let numberArr = arr.map((s) => +s);
        numberArr[1]--;
        if (
          numberArr[0] > 0 &&
          numberArr[1] >= 0 &&
          numberArr[1] < 12 &&
          !isNaN(new Date(numberArr[2], numberArr[1], 0).getTime()) &&
          numberArr[0] <= new Date(numberArr[2], numberArr[1] + 1, 0).getDate()
        ) {
          let date = new Date(numberArr[2], numberArr[1], numberArr[0]);
          if (!isNaN(date.getTime())) {
            this.currentDate = date;
          }
        }
      }
    },
    dayClick(day : Date) {
      this.currentDate = day;
      this.calendarVisible = false;
    }
  },
});
</script>

<style>
</style>