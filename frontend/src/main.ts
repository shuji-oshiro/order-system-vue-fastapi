/**
 * main.ts
 *
 * Bootstraps Vuetify and other plugins then mounts the App
 */

// Composables
import { createApp } from 'vue'
import { createPinia } from 'pinia'

// Components
import App from './App.vue'

// Plugins
import { registerPlugins } from '@/plugins'

// Styles
import 'unfonts.css'
import router from './router'

// アプリ作成
const app = createApp(App)
const pinia = createPinia()

// プラグイン登録（router, vuetifyなど）
registerPlugins(app)

// Pinia をアプリに登録
app.use(pinia)

// アプリをマウント
app.mount('#app')
