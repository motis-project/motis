<template>
  <InputField
    :labelName="$t.time"
    iconType="schedule"
    :showLabel="true"
    :initInputText="timeToDisplay"
    class="pure-u-1 pure-u-sm-9-24"
    :showArrows="true"
    @inputChanged="setTime"
    @decreaseClick="changeTime(-1)"
    @increaseClick="changeTime(1)"
    :showAutocomplete="false"
  />
</template>

<script lang="ts">
import { defineComponent } from "vue";
import InputField from "./InputField.vue";

export default defineComponent({
  name: "TimeInputField",
  components: {
    InputField,
  },
  data() {
    return {
      time: {} as Date,
    };
  },
  computed: {
    timeToDisplay: function (): String {
      let result = this.$ds.getTimeString(this.time.valueOf());
      this.$emit("timeChanged", this.time);
      return result;
    },
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
    },
    setTime(value: string) {
      let t = this.$ds.parseTime(value);
      if(t.valueOf()) {
        this.time = t;
      }
    },
  },
  created() {
    let currentTime = this.$ds.date;
    this.time = currentTime;
  },
});
</script>
