<template>
  <v-navigation-drawer 
    v-model="isNavigation"
    temporary
  >
  <Login />
  <!-- <ImportCSVBtn /> -->
  </v-navigation-drawer>  
</template>

<script setup lang="ts">
  import { ref } from 'vue'

  const isNavigation = ref<boolean>(false)

  </script>