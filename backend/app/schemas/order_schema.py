from .baseSchema import BaseSchema
from datetime import datetime
from typing import Optional

class MenuBase(BaseSchema):
    name: str
    price: int


class OrderIn(BaseSchema):
    seat_id: int
    menu_id: int
    order_cnt: int


class OrderOut(BaseSchema):
    id: int
    order_date: datetime # フォーマットされた日時文字列
    seat_id: int
    menu_id: int
    order_cnt: int
    menu: MenuBase
