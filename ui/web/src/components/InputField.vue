<template>
  <div>
    <div>
      <div v-if="showLabel" class="label">{{ labelName }}</div>
      <div class="gb-input-group">
        <div class="gb-input-icon">
          <i class="icon">{{ iconType }}</i>
        </div>
        <input class="gb-input" tabindex="1" @input="$emit('inputChanged', $event.target.value)" :value="initInputText">
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
    initInputText: String,
  },
});
</script>

<style>
</style>
