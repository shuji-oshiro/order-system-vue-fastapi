<template>
  <v-container>
    <v-text-field v-model="username" label="ユーザー名" />
    <v-text-field v-model="password" label="パスワード" type="password" />
    <v-btn @click="login">ログイン</v-btn>
  </v-container>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const username = ref('')
const password = ref('')

async function login() {
  try {
    const res = await axios.post('/login', {
      username: username.value,
      password: password.value,
    })
    localStorage.setItem('token', res.data.access_token)
    alert("ログイン成功！")
  } catch (e) {
    alert("ログイン失敗")
  }
}

axios.get('/protected', {
  headers: {
    Authorization: `Bearer ${localStorage.getItem('token')}`
  }
})
</script>
