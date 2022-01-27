<template>
  <div>
    <InputField
      :showLabel="true"
      :labelName="$t.date"
      iconType="event"
      :isTimeCalendarField="true"
      :showArrows="true"
      :initInputText="$ds.getDateString(currentDate.valueOf())"
      @inputChanged="onFieldInput"
      @focus="calendarVisible = true"
      @blur="inputBluredHandler"
      @decreaseClick="stopInterval(-1)"
      @increaseClick="stopInterval(1)"
      @decreaseMouseDown="mouseDown(-1)"
      @increaseMouseDown="mouseDown(1)"
      :showAutocomplete="false"
      :key="inputFieldKey"></InputField>
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
            'valid-day',
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
      inputFieldKey: 0
    };
  },
  watch: {
    currentDate: function (date: Date) {
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
        this.inputFieldKey++;
      }
    },
    onFieldInput(value: string) {
      let d = this.$ds.parseDate(value);
      if(d.valueOf()) {
        this.currentDate = d;
        this.$emit('dateChanged', this.currentDate);
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
    }
  },
});
</script>

<style>
</style>
