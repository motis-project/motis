<template>
  <div class="trip-search">
    <div class="header">
      <div id="trip-search-form">
        <div class="pure-g gutters">
          <InputField
            class="pure-u-1 pure-u-sm-1-2 train-nr"
            labelName="Zugnummer"
            iconType="train"
            :showLabel="true"
          ></InputField>
        </div>
        <div class="pure-g gutters">
          <Calendar class="pure-u-1 pure-u-sm-12-24 to-location"></Calendar>
          <TimeInputField></TimeInputField>
        </div>
      </div>
    </div>
    <div class="trips"></div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import Calendar from "../components/Calendar.vue";
import TimeInputField from "../components/TimeInputField.vue";
import InputField from "../components/InputField.vue";

export default defineComponent({
  name: "TrainSearch",
  components: {
    Calendar,
    TimeInputField,
    InputField,
  },
});
</script>
