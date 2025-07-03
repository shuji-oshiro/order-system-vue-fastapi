<template>
  <v-list-item
    title="注文履歴"
  ></v-list-item>        
  <v-list density="compact" nav>
    <v-list-item prepend-icon="mdi-view-dashboard" title="座席番号"></v-list-item>
    <v-list-item prepend-icon="mdi-forum" title="ご注文金額"></v-list-item>
  </v-list>
  <v-data-table-virtual
    :headers="headers"
    :items="orders"
    fixed-header
  >
  </v-data-table-virtual>
</template>

<script setup lang="ts">
  import axios from 'axios'
  import { ref, onMounted, watch, computed } from 'vue'
  import { AlertType } from '@/types/enums'
  import { CommonEventStore,UseEventStore } from '@/stores/eventStore'
  import type { OrderOut } from '@/types/orderTypes'  
  import type { DataTableHeader } from 'vuetify'

  const rawOrders  = ref<OrderOut[]>([]) // 注文リスト
  const commonEventStore = CommonEventStore()
  const useEventOrder = UseEventStore()
  // 合計金額付きの加工済みリスト
  const orders = computed(() =>
    rawOrders.value.map(order => ({
      id: order.id,
      order_date: order.order_date,
      seat_id: order.seat_id,
      menu_id: order.menu_id,
      order_cnt: order.order_cnt,
      menu_name: order.menu?.name ?? '',
      menu_price: order.menu?.price ?? 0,
      total: order.menu?.price ? order.menu.price * order.order_cnt : 0
    }))
  )

  const headers: DataTableHeader[] = [
    { title: '商品名', align: 'start', key: 'menu_name' },
    { title: '単価', align: 'end', key: 'menu_price' },
    { title: '注文数', align: 'end', key: 'order_cnt' },
    { title: '合計金額', align: 'end', key: 'total', sortable: true}
  ]


  // 注文情報を取得する関数
  // 注文情報は、座席IDを指定して取得する
  async function getOrderInfo(orderId: number = 1) {
    try {
      // シート単位の現在の注文状況を取得
      const response = await axios.get(`http://localhost:8000/order/${orderId}`)
      rawOrders.value = response.data || []

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
    () => {
      getOrderInfo();
    }
  )

  // 注文情報が更新された状態を検知し、注文情報を再取得する
  watch(
    () => useEventOrder.updateOrderAction.timestamp,
    () => {
      if (useEventOrder.updateOrderAction.updateflg) {
        getOrderInfo()
      }
    }
  ) 
</script>