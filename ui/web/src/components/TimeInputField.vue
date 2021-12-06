<template>
  <InputField
    labelName="Uhrzeit"
    iconType="schedule"
    :showLabel="true"
    :initInputText="timeToDisplay"
    class="pure-u-1 pure-u-1 pure-u-sm-9-24"
    :showArrows="true"
    @inputChanged="changeTime"
    @decreaseClick="setTime(-1)"
    @increaseClick="setTime(1)"
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
      return this.time.getHours() + ":" + ("0" + this.time.getMinutes()).slice(-2);
    },
  },
  methods: {
    setTime(change: number) {
      this.time = new Date(
        this.time.getFullYear(),
        this.time.getMonth(),
        this.time.getDay(),
        this.time.getHours() + change,
        this.time.getMinutes()
      );
    },
    changeTime(value: string) {
      let arr = value.split(":");
      if (arr.length === 2 && arr[1].length === 2) {
        let numberArr = arr.map((s) => +s);
        if (
          numberArr[0] >= 0 && numberArr[0] < 24 &&
          numberArr[1] >= 0 && numberArr[1] < 60  
        ) {
          let date = new Date(new Date().getFullYear(), 
                              new Date().getMonth(), 
                              new Date().getDay(), 
                              numberArr[0], numberArr[1]);
          if (!isNaN(date.getTime())) {
            this.time = date;
          }
        }
      }
    },
  },
  created() {
    let currentTime = new Date();
    this.time = currentTime;
  },
});
</script>
