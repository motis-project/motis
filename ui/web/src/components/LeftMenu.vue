<template>
    <InputField labelName="Start" iconType="place"/>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import InputField from '../components/InputField.vue';

export default defineComponent({
  name: 'LeftMenu',
  components: {
    InputField,
  }
});
</script>

<style>
</style>
