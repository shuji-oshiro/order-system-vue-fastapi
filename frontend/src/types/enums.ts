import type Monitoring from "@/components/common/Monitoring.vue";

  
  // ナビゲーションバーの種類を定義
  export enum NavigationType {
      History = 'history',
      Category = 'category',
      Order = 'order',
      Maintenance = 'maintenance',
      Monitoring = 'monitoring'
    }

  // アラートの種類を定義
  export enum AlertType {
    Success = 'success',
    Error = 'error',
    Info = 'info',
    Warning = 'warning'
  }