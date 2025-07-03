import { defineConfig } from 'vite'
import { fileURLToPath, URL } from 'node:url'
import Vue from '@vitejs/plugin-vue'
import Vuetify, { transformAssetUrls } from 'vite-plugin-vuetify'
import Fonts from 'unplugin-fonts/vite'
import Components from 'unplugin-vue-components/vite'
import VueRouter from 'unplugin-vue-router/vite'

export default defineConfig({
  root: '.', // frontend/ 以下にあるなら省略可
  plugins: [
    VueRouter({ dts: 'src/typed-router.d.ts' }),
    Vue({ template: { transformAssetUrls } }),
    Vuetify({
      autoImport: true,
      styles: { configFile: 'src/styles/settings.scss' },
    }),
    Components({
      dirs: ['src/components'],
      extensions: ['vue', 'tsx'],
      deep: true,
      dts: 'src/components.d.ts'
    }),
    Fonts({
      fontsource: {
        families: [
          { name: 'Roboto', weights: [100, 300, 400, 500, 700, 900], styles: ['normal', 'italic'] },
        ],
      },
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
    extensions: ['.js', '.json', '.ts', '.vue', '.tsx'],
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: ['.ngrok-free.app'],
  },
  css: {
    preprocessorOptions: {
      sass: { api: 'modern-compiler' },
      scss: { api: 'modern-compiler' },
    },
  },
})
