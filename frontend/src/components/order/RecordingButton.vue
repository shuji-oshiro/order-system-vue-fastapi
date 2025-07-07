<template>
  <v-btn
    @click="openDialog"
    style="min-height: 48px; height: 48px; padding-top: 0; padding-bottom: 0; align-items: center;"
    class="d-flex flex-column justify-center"
  >
    <v-icon size="28" class="mb-0">
      {{ isRecording ? 'mdi-record-rec' : 'mdi-microphone' }}
    </v-icon>
    <span style="font-size: 12px; line-height: 1; margin-top: 0;">
      {{ isRecording ? '停止' : '注文' }}
    </span>
  </v-btn>

  <v-dialog v-model="dialog" max-width="400" persistent>
    <v-card>
      <v-card-title class="text-h6">
        {{ dialogTitle }}
      </v-card-title>
      <v-card-text>
        <div v-if="isRecording">
          録音中...（{{ elapsed }}秒）
        </div>
        <div v-else-if="!isRecording">
          <div>認識結果: {{ reco_text }}</div>

          <div v-if="isCorect">料理名: {{ match_text }}</div>
          <div v-else>料理名が認識できませんでした。</div>
        </div>
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn
          v-if="isRecording"
          color="error"
          @click="stopRecording"
        >停止</v-btn>
        <v-btn
          v-else-if="showOrderButton"
          color="primary"
          @click="orderAndClose"
        >注文</v-btn>
        <v-btn
          v-else-if="!isRecording"
          color="secondary"
          @click="closeDialog"
        >閉じる</v-btn>
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>

<script setup lang="ts">

import { ref, onUnmounted, computed } from 'vue'
import type { VoiceResult } from '@/types/menuTypes'
import type { MatchMenu, MenuOut } from '@/types/menuTypes'
import axios from 'axios'

const dialog = ref(false)
const isRecording = ref(false)
const isCorect = ref(false)
const reco_text = ref('')
const match_text = ref('')
const elapsed = ref(0)

let timer: number | null = null
let recorder: MediaRecorder | null = null
let chunks: Blob[] = []

const emit = defineEmits<{
  (e: 'recorded', reco_text: string, match_text:string, blob:Blob): void
  (e: 'error', message: string): void
}>()

const dialogTitle = computed(() =>
  isRecording.value ? '音声録音中' : '認識結果'
)

const showOrderButton = computed(() =>
  !isRecording.value && match_text.value
)

function openDialog() {
  dialog.value = true
  startRecording()
}

function closeDialog() {
  dialog.value = false
  resetState()
}

function resetState() {
  isRecording.value = false
  reco_text.value = ''
  match_text.value = ''
  elapsed.value = 0
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}

async function startRecording() {
  isRecording.value = true
  reco_text.value = ''
  match_text.value = ''
  elapsed.value = 0
  chunks = []

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  recorder.ondataavailable = (e) => chunks.push(e.data)
  
  // 録音停止時の処理
  recorder.onstop = async () => {
    const blob = new Blob(chunks, { type: 'audio/webm' })
    chunks = []
    // サーバーへ送信
    const formData = new FormData()
    formData.append('file', blob, 'recording.webm')
    try {
      const response = await axios.post('/api/voice', formData)
      const voiceResult = response.data

      reco_text.value = voiceResult.reco_text
      const match_menus = voiceResult.match_menus as MatchMenu[]

      let match_menus_text = ""
      match_menus.forEach((match_menu) => {
        const menu = match_menu.menu as MenuOut
        const val = "商品名：" + menu.name + " スコア：" + match_menu.score
        match_menus_text += val + "\n"
      })
      if (match_menus_text === "") {
        emit('error', '料理名が認識できませんでした')
        isCorect.value = false
        return;
      }
      match_text.value = match_menus_text

      isCorect.value = true
      emit('recorded', reco_text.value, match_text.value, blob)
    } catch (err) {
      
      emit('error', '通信エラーが発生しました')
    }finally {
      isRecording.value = false
    }
  }
  // 録音開始 最大5秒間録音
  recorder.start()
  timer = setInterval(() => {
    elapsed.value += 1
    if (elapsed.value >= 5) {
      stopRecording()
    }
  }, 1000)
}

// 録音停止
function stopRecording() {
  if (recorder?.state === 'recording') {
    recorder.stop()
    //isRecording.value = false
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  }
}

function orderAndClose() {
  // TODO：注文処理を実装
  closeDialog()
}

onUnmounted(() => {
  if (timer) clearInterval(timer)
  if (recorder && recorder.state === 'recording') recorder.stop()
})
</script>
