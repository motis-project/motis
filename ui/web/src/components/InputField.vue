<template>
    <div>
        <div>
            <div v-show=show class="label">{{ labelName }}</div>
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
    <InputField show: true labelName="Start" iconType="place"/>
    to not display the labelName remove show: true
-->


<script lang="ts">
import { defineComponent } from 'vue';

export default defineComponent({
  name: 'InputField',
  props: {
      labelName: String,
      iconType: String,
      show: Boolean,
  }
});
</script>

<style>
</style>