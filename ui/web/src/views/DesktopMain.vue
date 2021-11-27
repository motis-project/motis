<template>
  <br><br>
  <InputField labelName="Start" iconType="place" className="pure-u-1 pure-u-sm-12-24 from location"/>
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import LeftMenu from '../components/LeftMenu.vue'
import InputField from '../components/InputField.vue'

export default defineComponent({
  name: 'DesktopMain',
  components:
  {
    LeftMenu,
    InputField
  }
});
</script>

<style>
</style>
