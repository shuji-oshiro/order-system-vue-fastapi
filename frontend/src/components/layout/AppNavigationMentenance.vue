<template>
  <v-navigation-drawer 
    v-model="isNavigation"
    temporary
  >
  <Login @login-status-changed="handleLoginStatusChange" />
  <ImportCSVBtn v-if="isLoggedIn" />
  <v-divider v-if="isLoggedIn" class="my-4"></v-divider>
  </v-navigation-drawer>  
</template>

<script setup lang="ts">
  import { ref } from 'vue'

  const isNavigation = ref<boolean>(false)
  const isLoggedIn = ref<boolean>(false)

  // ログイン状態の変更を処理
  function handleLoginStatusChange(loginStatus: boolean) {
    isLoggedIn.value = loginStatus
  }
</script>