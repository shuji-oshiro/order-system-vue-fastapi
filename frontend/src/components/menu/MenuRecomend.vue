<template>
  <v-list-item
    title="おすすめメニュー"
  ></v-list-item>        
  <div style="max-height: 90vh; overflow-y: auto; overflow-x: hidden;">
    <v-row dense>
      <v-col
      v-for="menu in recomend_menus"
      :key="menu.id"
      cols="12"
      >
      <FoodCard :menu="menu" :onClick="() => selectMenu(menu)" />
      </v-col>
    </v-row>
  </div>
</template>

<script setup lang="ts">
  import axios from 'axios'
  import { ref, onMounted, watch } from 'vue'
  import { AlertType } from '@/types/enums'
  import { CommonEventStore, UseEventStore } from '@/stores/eventStore'
  import type { MenuOut } from '@/types/menuTypes'

  const recomend_menus  = ref<MenuOut[]>([]) // おすすめメニューのリストを管理する変数
  const commonEventStore = CommonEventStore()
  const useEventStore = UseEventStore()

  const selectMenu = (menu: MenuOut) => {
    useEventStore.triggerMenuSelectAction(menu)
  }


  //おすすめメニューを取得する関数
  // この関数は、コンポーネントの初期化時に呼び出され、サーバーからおすすめメニューを取得します。
  async function getRecomendMenu() {
    try {
      // シート単位の現在の注文状況を取得
      const response = await axios.get('/api/menu/category/1')
      recomend_menus.value = response.data || []

    } catch (error) {
      if (axios.isAxiosError(error)) {
        const errorMessage = error.response?.data?.detail || 'サーバーからの応答がありません'
        commonEventStore.EventAlertInformation(AlertType.Error, "注文履歴情報の取得中にエラーが発生しました", errorMessage)
      } else {
        commonEventStore.EventAlertInformation(AlertType.Error, "注文履歴情報の取得中にエラーが発生しました", '予期しないエラーが発生しました')
      }
    }
  }

  // 初期化時に注文情報を取得
  onMounted(
    getRecomendMenu
  )

  // // 注文情報が更新されたときに再取得
  // watch(
  //   () => store.updateOrderAction.timestamp,
  //   () => {
  //     if (store.updateOrderAction.timestamp) {
  //       getOrderInfo()
  //     }
  //   }
  // ) 
</script>