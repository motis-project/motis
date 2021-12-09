<template>
  <div class="loading center">
    <div class="spinner">
      <div class="bounce1"></div>
      <div class="bounce2"></div>
      <div class="bounce3"></div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from 'vue'

export default defineComponent({
    name: "LoadingBar"
})
</script>


<style scoped>
.center {
    margin: auto;
}
</style>