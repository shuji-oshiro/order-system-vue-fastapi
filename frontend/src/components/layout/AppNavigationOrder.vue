<template>
  <v-navigation-drawer 
    v-model="isNavigation"
    temporary
  >
    <OrderConfirm />
    <MenuRecomend v-if="isMenuRecomend" />
  </v-navigation-drawer>  
</template>

<script setup lang="ts">
  import { ref } from 'vue'

  const isNavigation = ref<boolean>(false)
  const isMenuRecomend = ref<boolean>(false)


  </script>