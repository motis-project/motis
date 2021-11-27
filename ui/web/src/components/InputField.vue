<template>
    <div :class="className">
        <div>
            <div class="label">{{ labelName }}</div>
            <div class="gb-input-group">
                <div class="gb-input-icon">
                    <i class="icon">{{ iconType }}</i>
                </div>
                <input class="gb-input" tabindex="1">
            </div>
        </div>
    </div>
</template>


<!-- How to use this component:
    <InputField labelName="Start" iconType="place"/>
-->


<script lang="ts">
import { defineComponent } from 'vue';

export default defineComponent({
  name: 'InputField',
  props: {
      labelName: String,
      iconType: String,
      className: String
  }
});
</script>

<style>
</style>