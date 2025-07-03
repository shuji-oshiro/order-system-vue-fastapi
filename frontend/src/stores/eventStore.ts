import { defineStore } from 'pinia'
import type { MenuOut } from '@/types/menuTypes'
import { AlertType } from '@/types/enums'

// 共通イベントストア
// エラー、情報、警告イベントを管理するためのストア
export const CommonEventStore = defineStore('eventOrder', {

  state: () => ({
    // エラーイベント
    AlertInformation: {
      alertType: AlertType.Error,
      message: '',
      detail: '',
      source: '',
      timestamp: null as number | null,
    },
  }),
  actions: {
    EventAlertInformation(alertType: AlertType, message: string, detail: string = '', source: string = 'EventStore') {
      this.AlertInformation = {
        alertType,
        message,
        detail,
        source,
        timestamp: Date.now(),
      }
    },
  }
})


export const UseEventStore = defineStore('event', {
  state: () => ({
    // 注文選択イベント
    menuSelectAction: {
      source: '',
      menu: null as MenuOut | null,
      timestamp: null as number | null
    },

    // メニューカテゴリ選択イベント
    selectCategoryAction: {
      source: '',
      categoryId: null as number | null,
      timestamp: null as number | null
    },

    //注文更新イベント
    updateOrderAction: {
      source: '',
      updateflg: false,
      timestamp: null as number | null
    },

  }),

  actions: {
    triggerMenuSelectAction(menu: MenuOut | null, source = 'MenuList') {
      this.menuSelectAction = {
        source,
        menu,
        timestamp: Date.now()
      }
    },

    triggerSelectCategoryAction(categoryId: number, source = 'CategoryList') {
      this.selectCategoryAction = {
        source,
        categoryId,
        timestamp: Date.now()
      }
    },

    triggerUpdateOrderAction(flg: boolean, source = 'FoodMenusList') {
      this.updateOrderAction = {
        source,
        updateflg: flg,
        timestamp: Date.now()
      }
    }, 
  }
})
