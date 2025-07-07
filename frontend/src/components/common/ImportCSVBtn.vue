<template>
  <v-file-input
    v-model="selectedFile"
    label="CSVファイルを選択"
    accept=".csv"
    prepend-icon="mdi-import"
    @change="onFileChange"
    hide-details
    dense
  ></v-file-input>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import axios from 'axios'
import { AlertType } from '@/types/enums'
import { CommonEventStore } from '@/stores/eventStore'
const commonEventStore = CommonEventStore()

const selectedFile = ref<File | null>(null)

const onFileChange = async () => {
  if (!selectedFile.value) return
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  await importMenus(formData)
  // ファイル選択状態をリセットしたい場合
  selectedFile.value = null
}

async function importMenus(formData: FormData) {
  try {
    const response = await axios.post('/api/menu', formData)
    commonEventStore.EventAlertInformation(AlertType.Success, "メニューの一括更新が完了しました")
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const errorMessage = error.response?.data?.detail || 'サーバーからの応答がありません'
      commonEventStore.EventAlertInformation(AlertType.Error, "メニューの一括更新中にエラーが発生しました", errorMessage)
    } else {
      commonEventStore.EventAlertInformation(AlertType.Error, 'メニュー更新中に予期しないエラーが発生しました')
    }
  }
}
</script>
