<template>
  <v-list-item
      title="メニューカテゴリ"
  ></v-list-item>
  <v-divider></v-divider>
  
  <v-list density="compact" nav>
    <v-card
        v-for="(category, index) in categories"
        :key="category.id"
        :title="category.name"
        :subtitle="category.description"
        @click="selectCategory(category.id)"
      ></v-card>
  </v-list>
</template>

<script setup lang="ts">
  import axios from 'axios'
  import { ref, onMounted } from 'vue'
  import { AlertType } from '@/types/enums'
  import { UseEventStore, CommonEventStore } from '@/stores/eventStore'
  import type { MenuCategory } from '@/types/menuTypes'
  const useEventStore = UseEventStore()
  const commonEventStore = CommonEventStore()
  const categories = ref([] as MenuCategory[])

  // カテゴリが選択された時の処理
  // カテゴリIDを引数に受け取り、Pinia のアクションを呼び出す
  function selectCategory(id: number) {
    useEventStore.triggerSelectCategoryAction(id) 
  }

  // カテゴリ情報を取得する関数
  async function getCategoryInfo() {
    try {
      // カテゴリ単位の現在のメニュー状況を取得
      const response = await axios.get(`/api/category`)
      categories.value = response.data || []

    } catch (error) {
      if (axios.isAxiosError(error)) {
        // Axios エラーの場合
          commonEventStore.EventAlertInformation(AlertType.Error, `カテゴリ情報の取得に失敗しました: ${error.message}`)
        } else {
          commonEventStore.EventAlertInformation(AlertType.Error, 'カテゴリ情報の取得中に予期しないエラーが発生しました')
      }
    }
  }

  // コンポーネントがマウントされた時にカテゴリ情報を取得
  onMounted(() => {
    getCategoryInfo()
  })

</script>