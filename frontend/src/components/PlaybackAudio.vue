<template>

  <audio v-if="audioUrl" :src="audioUrl" controls />

</template>

<script setup lang="ts">
defineProps<{
  audioUrl: string | null
}>()
</script>
