<template>
  <v-bottom-navigation>
    <v-btn @click="on_click(NavigationType.Order)">
    <v-icon>mdi-cart-check</v-icon>
    <span>注文待機</span>
    </v-btn>
    <v-btn @click="on_click(NavigationType.History)">
    <v-icon>mdi-history</v-icon>
    <span>注文履歴</span>
    </v-btn>
    <v-btn @click="on_click(NavigationType.Category)">
    <v-icon>mdi-silverware</v-icon>
    <span>メニュー</span>
    </v-btn>   
    <v-btn @click="on_click(NavigationType.Maintenance)">
    <v-icon>mdi-tools</v-icon>
    <span>メンテナンス</span>
    </v-btn>   

    <v-btn @click="on_click(NavigationType.Monitoring)">
    <v-icon>mdi-monitor-eye</v-icon>
    <span>監視</span>
    </v-btn>

    <RecordingButton />
  </v-bottom-navigation>
</template>
<script setup lang="ts">
  import { NavigationType } from '@/types/enums'
  
  const emit = defineEmits<{
    (e: 'click', value: NavigationType): void
  }>()
 
  //呼び出しナビゲーションボタン押下時
  function on_click(target: NavigationType) {
    emit('click', target)
  }
  
 

</script>
