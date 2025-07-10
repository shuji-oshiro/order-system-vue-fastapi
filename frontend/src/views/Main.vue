<template>
  <v-card>
    <v-layout>
      <AppNavigationCategory v-model="isNavigationCategory"/>
      <AppNavigationOrder v-model="isNavigationOrder"/>
      <AppNavigationHistory v-model="isNavigationHistory"/>
      <AppNavigationMentenance v-model="isNavigationMaintenance" />
      <v-main style="height: 100vh;" >  
        <!-- ダイアログでアラートを表示し、他の操作をブロック -->
        <v-dialog v-model="showAlert" persistent width="400">
          <v-card>
            <v-alert
              :title="title"
              :text="message"
              :type="alertType"
              variant="tonal"
            ></v-alert>
            <v-card-actions>
              <v-spacer />
              <v-btn color="primary" @click="showAlert = false">閉じる</v-btn>
            </v-card-actions>
          </v-card>
        </v-dialog>
        <v-dialog v-model="showMonitoring" max-width="1200px" persistent>
          <div class="d-flex justify-end pa-3">
            <v-btn @click="showMonitoring = false" color="primary">閉じる</v-btn>
          </div>
          <MonitoringDashboard />
        </v-dialog>
        <MenuWindow @click="selectMenu" />
        <AppBottomNavigation @click="showNavigation" />
      </v-main>
    </v-layout>
  </v-card>
</template>
<script setup lang="ts">
  import { ref, watch} from 'vue'
  import { NavigationType, AlertType } from '@/types/enums'
  import type { MenuOut } from '@/types/menuTypes'
  import { CommonEventStore, UseEventStore } from '@/stores/eventStore'
  const useEventStore = UseEventStore()
  const commonEventStore = CommonEventStore()

  const isNavigationCategory = ref<boolean>(false)
  const isNavigationOrder = ref<boolean>(false)
  const isNavigationHistory = ref<boolean>(false)
  const isNavigationMaintenance = ref<boolean>(false)
  const showMonitoring = ref<boolean>(false)

  const showAlert = ref(false)
  const alertType = ref<AlertType>(AlertType.Error)
  const title = ref<string>('')
  const message = ref<string>('')

  // ナビゲーションバーよりカテゴリが選択された時、またはメニューがインポートされた時の処理を監視
  watch(
    () => commonEventStore.AlertInformation.timestamp,
    () => {
      if (commonEventStore.AlertInformation.timestamp) {
        showAlert.value = true
        title.value = commonEventStore.AlertInformation.message
        message.value = commonEventStore.AlertInformation.detail
        alertType.value = commonEventStore.AlertInformation.alertType
      }
    }
  )

  function selectMenu(menu: MenuOut) {
    // メニューが選択されたことを通知
    useEventStore.triggerMenuSelectAction(menu)
    // ナビゲーションの状態を更新
    showNavigation(NavigationType.Order)
  }


  // ナビゲーションバーよりカテゴリが選択された時、またはメニューがインポートされた時の処理を監視
  function showNavigation(target: NavigationType) {

    isNavigationCategory.value = false
    isNavigationOrder.value = false
    isNavigationHistory.value = false
    isNavigationMaintenance.value = false
    showMonitoring.value = false

    if (target === NavigationType.Category) {
      isNavigationCategory.value = true
    } else if (target === NavigationType.Order) {
      isNavigationOrder.value = true
    } else if (target === NavigationType.History) {
      isNavigationHistory.value = true
    } else if (target === NavigationType.Maintenance) {
      isNavigationMaintenance.value = true
    } else if (target === NavigationType.Monitoring) {
      showMonitoring.value = true
    }
  }

</script>