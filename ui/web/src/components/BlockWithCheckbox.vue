<template>
  <fieldset class="mode walk">
    <legend class="mode-header">
      <label>
        <input type="checkbox" :checked="isChecked" @input="$emit('isCheckedChanged', $event.target.checked)"/>
        {{title}}
      </label>
    </legend>
    <slot></slot>
  </fieldset>
</template>

<script lang="ts">
import { defineComponent, PropType } from 'vue';

export default defineComponent({
  name: "BlockWithCheckbox",
  props: {
    title: {
      type: String as PropType<String>,
      required: true
    },
    isChecked: {
      type: Boolean as PropType<Boolean>,
      required: true
    }
  }
})
</script>

<style>
</style>