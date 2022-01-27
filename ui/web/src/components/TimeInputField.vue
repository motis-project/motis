<template>
  <InputField
    :labelName="$t.time"
    iconType="schedule"
    :showLabel="true"
    :initInputText="timeToDisplay"
    class="main-gutter time-gutter"
    :isTimeCalendarField="true"
    :showArrows="true"
    @inputChanged="setTime"
    @decreaseClick="changeTime(-1)"
    @increaseClick="changeTime(1)"
    :showAutocomplete="false"
    :key="inputFieldKey"
    @blur="inputFieldKey++"></InputField>
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
      inputFieldKey: 0
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
    setTime(value: string) {
      let t = this.$ds.parseTime(value);
      if(t.valueOf()) {
        this.time = t;
      }
      this.$emit("timeChanged", this.time);
    },
  },
});
</script>
