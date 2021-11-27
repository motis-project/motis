<template>
  <div>
    <div>
      <div v-if="showLabel" class="label">{{ labelName }}</div>
      <div class="gb-input-group">
        <div class="gb-input-icon">
          <i class="icon">{{ iconType }}</i>
        </div>
        <input class="gb-input" v-if="id === 'start' || id === 'dest'" :id="id" tabindex="1" @change="sendChangedText" v-model="inputText">
        <input class="gb-input" v-else-if="id === 'time'" :id="id" tabindex="1" @change="sendChangedText" v-model="time">
      </div>
    </div>
  </div>
</template>


<!-- How to use this component:
    <InputField :showLabel=true labelName="Start" iconType="place"/>
    to not display the labelName set :showLabel to false
-->


<script lang="ts">
import { defineComponent } from "vue";

export default defineComponent({
  name: "InputField",
  props: {
    labelName: String,
    iconType: String,
    showLabel: Boolean,
    id: String,
  },
  data() {
    return {
      inputText: '',
      time: (new Date).getHours() + ":" + ("0" + (new Date).getMinutes()).slice(-2),
    }
  },
  methods: {
    sendChangedText() {
      if(this.labelName === 'Start')
        this.$emit('getStartInput', (<HTMLInputElement>document.getElementById("start")).value);
      else
        this.$emit('getDestInput', (<HTMLInputElement>document.getElementById("dest")).value);
    },
  }
});
</script>

<style>
</style>
