
// シート単位の注文履歴
export type MenuBase = {
  name: string
  price: number
}

export type OrderOut = {
  id: number
  order_date: string // datetime → ISO 8601 形式の文字列（例："2025-06-25T12:34:56"）
  seat_id: number
  menu_id: number
  order_cnt: number
  menu: MenuBase
}
