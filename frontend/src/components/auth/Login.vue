<template>
  <v-container>
    <!-- ログイン済みの場合の表示 -->
    <v-card v-if="isLoggedIn" class="mb-4" color="success" variant="tonal">
      <v-card-title>ログイン中</v-card-title>
      <v-card-text>
        <p><strong>ユーザー名:</strong> {{ userInfo?.username }}</p>
        <p><strong>ログイン時刻:</strong> {{ new Date(userInfo?.loginTime).toLocaleString() }}</p>
        <p><strong>トークン有効期限:</strong> {{ new Date(tokenExpiry).toLocaleString() }}</p>
        <p><strong>残り時間:</strong> 
          <v-chip :color="remainingTime > 10 ? 'success' : 'warning'" size="small">
            {{ formatRemainingTime }}
          </v-chip>
        </p>
        <p><strong>トークン状態:</strong> 
          <v-chip :color="isTokenValid ? 'success' : 'error'" size="small">
            {{ isTokenValid ? '有効' : '期限切れ' }}
          </v-chip>
        </p>
      </v-card-text>
      <v-card-actions class="d-flex flex-column align-start">
        <v-btn @click="logout" color="error" variant="outlined" class="mb-2">ログアウト</v-btn>
        <v-btn @click="extendToken" color="info" variant="outlined" class="mb-2">
          トークン延長（+30秒）
        </v-btn>
        <v-btn @click="testProtectedAPI" color="purple" variant="outlined">
          認証APIテスト
        </v-btn>
      </v-card-actions>
    </v-card>

    <!-- ログインフォーム -->
    <v-card v-else>
      <v-card-title>ログイン</v-card-title>
      <v-card-text>
        <v-text-field v-model="username" label="ユーザー名" />
        <v-text-field v-model="password" label="パスワード" type="password" />
      </v-card-text>
      <v-card-actions>
        <v-btn @click="login" color="primary" class="mb-4">ログイン</v-btn>
      </v-card-actions>
    </v-card>
    
    <v-divider class="my-4"></v-divider>
    
    <v-btn @click="createTestUser" color="secondary" variant="outlined">
      テスト用ユーザー作成
    </v-btn>
  </v-container>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import axios from 'axios'

// Emitの定義
const emit = defineEmits(['login-status-changed'])

const username = ref('')
const password = ref('')
const isLoggedIn = ref(false)
const userInfo = ref(null)
const tokenExpiry = ref(null)
const remainingTime = ref(0)
const checkInterval = ref(null)

// ログイン状態の変更を監視してイベントを発行
watch(isLoggedIn, (newValue) => {
  emit('login-status-changed', newValue)
}, { immediate: true })

// トークンの有効性をチェック
const isTokenValid = computed(() => {
  if (!tokenExpiry.value) return false
  return remainingTime.value > 0
})

// 残り時間を計算（秒）
const getRemainingSeconds = computed(() => {
  if (!tokenExpiry.value) return 0
  const remaining = Math.floor((new Date(tokenExpiry.value) - new Date()) / 1000)
  return Math.max(0, remaining)
})

// 残り時間をフォーマット
const formatRemainingTime = computed(() => {
  const seconds = remainingTime.value
  if (seconds <= 0) return '期限切れ'
  
  const minutes = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${minutes}分${secs}秒`
})

// ログイン状態を初期化
onMounted(() => {
  checkLoginStatus()
  // 初期状態でもイベントを発行
  emit('login-status-changed', isLoggedIn.value)
})

// リアルタイムで残り時間を更新
function startTokenTimer() {

  if (checkInterval.value) {
    clearInterval(checkInterval.value)
  }
  
  checkInterval.value = setInterval(() => {
    const currentTime = new Date()
    const expiryTime = new Date(tokenExpiry.value)
    const remainingSeconds = Math.floor((expiryTime - currentTime) / 1000)
    
    remainingTime.value = Math.max(0, remainingSeconds)
    
    // トークンが期限切れの場合は自動ログアウト
    if (remainingSeconds <= 0 && isLoggedIn.value) {
      alert('トークンが期限切れになりました。自動ログアウトします。')
      logout()
    }
  }, 1000) // 1秒ごとに更新
}

// タイマーを停止
function stopTokenTimer() {
  if (checkInterval.value) {
    clearInterval(checkInterval.value)
    checkInterval.value = null
  }
}

// ログイン状態をチェック
function checkLoginStatus() {
  const token = localStorage.getItem('token')
  const expiry = localStorage.getItem('tokenExpiry')
  const user = localStorage.getItem('userInfo')
  
  if (token && expiry && user) {
    if (new Date() < new Date(expiry)) {
      isLoggedIn.value = true
      tokenExpiry.value = expiry
      userInfo.value = JSON.parse(user)
      startTokenTimer() // タイマーを開始
    } else {
      // トークンが期限切れの場合はクリア
      logout()
    }
  }
}

// JWTトークンから有効期限を取得
function getTokenExpiry(token) {
  try {
    // JWTトークンのペイロード部分をデコード
    const payload = JSON.parse(atob(token.split('.')[1]))
    console.log('JWTペイロード:', payload)
    
    // JWTの'exp'フィールドから有効期限を取得（Unix timestamp）
    if (payload.exp) {
      const expiry = new Date(payload.exp * 1000) // Unix timestampをミリ秒に変換
      console.log('JWTから取得した有効期限:', expiry)
      return expiry
    } else {
      // 'exp'フィールドがない場合は現在時刻から1時間後をデフォルトに設定
      console.warn('JWTに有効期限(exp)が含まれていません。デフォルト値を使用します。')
      const defaultExpiry = new Date(Date.now() + 60 * 60 * 1000) // 1時間後
      console.log('デフォルト有効期限:', defaultExpiry)
      return defaultExpiry
    }
  } catch (error) {
    console.error('JWTトークンの解析に失敗:', error)
    // エラーの場合は現在時刻から1時間後をフォールバックとして設定
    const fallbackExpiry = new Date(Date.now() + 60 * 60 * 1000) // 1時間後
    console.log('フォールバック有効期限:', fallbackExpiry)
    return fallbackExpiry
  }
}

// ログアウト
function logout() {
  stopTokenTimer() // タイマーを停止
  localStorage.removeItem('token')
  localStorage.removeItem('tokenExpiry')
  localStorage.removeItem('userInfo')
  isLoggedIn.value = false
  userInfo.value = null
  tokenExpiry.value = null
  username.value = ''
  password.value = ''
}

async function login() {
  try {
    // フォームデータとして送信（FastAPIのOAuth2PasswordRequestFormに対応）
    const formData = new FormData()
    formData.append('username', username.value)
    formData.append('password', password.value)

    const res = await axios.post('/api/auth/login', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    
    const token = res.data.access_token
    const expiry = getTokenExpiry(token)
    const user = {
      username: username.value,
      loginTime: new Date().toISOString()
    }
    
    // トークン情報を保存
    localStorage.setItem('token', token)
    localStorage.setItem('tokenExpiry', expiry.toISOString())
    localStorage.setItem('userInfo', JSON.stringify(user))
    
    // 状態を更新
    isLoggedIn.value = true
    tokenExpiry.value = expiry.toISOString()
    userInfo.value = user
    startTokenTimer() // タイマーを開始
    
    alert("ログイン成功！")
  } catch (e) {
    if (e.response && e.response.data && e.response.data.detail) {
      alert(`ログイン失敗: ${e.response.data.detail}`)
    } else {
      alert("ログイン失敗")
    }
  }
}

async function createTestUser() {
  try {
    const testUsername = 'testuser'
    const testPassword = 'testpass123'
    
    const userData = {
      username: testUsername,
      password: testPassword,
      email: 'test@example.com'
    }
    
    const res = await axios.post('/api/auth/create_user', userData, {
      headers: {
        'Content-Type': 'application/json'
      }
    })
    
    if (res.data.success) {
      alert(`テストユーザーを作成しました！\nユーザー名: ${testUsername}\nパスワード: ${testPassword}`)
      // 作成後、フォームに自動入力
      username.value = testUsername
      password.value = testPassword
    } else {
      alert(`ユーザー作成に失敗しました: ${res.data.message}`)
    }
  } catch (e) {
    if (e.response && e.response.data && e.response.data.message) {
      alert(`ユーザー作成失敗: ${e.response.data.message}`)
    } else {
      alert("ユーザー作成に失敗しました")
    }
  }
}

// テスト用: トークン延長機能（実際のJWT有効期限に基づいて延長）
function extendToken() {
  const currentToken = localStorage.getItem('token')
  if (!currentToken) {
    alert('延長するトークンがありません')
    return
  }
  
  try {
    // 現在のJWTから有効期限を取得
    const payload = JSON.parse(atob(currentToken.split('.')[1]))
    if (payload.exp) {
      // 現在の有効期限から30分延長
      const currentExpiry = new Date(payload.exp * 1000)
      const newExpiry = new Date(currentExpiry.getTime() + 30 * 60 * 1000) // 30分延長
      
      localStorage.setItem('tokenExpiry', newExpiry.toISOString())
      tokenExpiry.value = newExpiry.toISOString()
      alert('トークンを30分延長しました！')
    } else {
      // expフィールドがない場合は現在時刻から30分後
      const newExpiry = new Date(Date.now() + 30 * 60 * 1000)
      localStorage.setItem('tokenExpiry', newExpiry.toISOString())
      tokenExpiry.value = newExpiry.toISOString()
      alert('トークンを30分延長しました！（現在時刻基準）')
    }
  } catch (error) {
    console.error('トークン延長エラー:', error)
    alert('トークンの延長に失敗しました')
  }
}

// 認証が必要なAPIを呼び出す例
function callProtectedAPI(endpoint = '/api/menu', method = 'GET', data = null) {
  const token = localStorage.getItem('token')
  if (!token || !isTokenValid.value) {
    alert('ログインが必要です')
    return Promise.reject('認証が必要です')
  }
  
  const config = {
    method,
    url: endpoint,
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json'
    }
  }
  
  if (data && (method === 'POST' || method === 'PUT')) {
    config.data = data
  }
  
  return axios(config)
}

// 認証APIのテスト関数
async function testProtectedAPI() {
  try {
    console.log('認証APIテスト開始')
    
    // メニュー一覧取得をテスト
    const response = await callProtectedAPI('/api/menu')
    console.log('APIレスポンス:', response.data)
    alert(`認証APIテスト成功！\nメニュー数: ${response.data.length || 0}件`)
    
  } catch (error) {
    console.error('認証APIテストエラー:', error)
    if (error.response?.status === 401) {
      alert('認証エラー: トークンが無効です')
    } else if (error.response?.status === 404) {
      alert('エンドポイントが見つかりません（正常な動作）')
    } else {
      alert(`APIエラー: ${error.message}`)
    }
  }
}

// Axiosインターセプターを設定してトークンを自動付与
axios.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token && isTokenValid.value) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// レスポンスインターセプターで401エラーを処理
axios.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      alert('認証が無効です。再ログインしてください。')
      logout()
    }
    return Promise.reject(error)
  }
)
</script>
