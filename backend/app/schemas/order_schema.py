from datetime import datetime
from typing import Optional
from pydantic import Field, field_validator
from .baseSchema import BaseSchema
from .menu_schema import MenuOut, MenuBase, CategoryBase

# class MenuBase(BaseSchema):
#     name: str
#     price: int


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


class OrderSchema(BaseSchema):
    id: int
    order_date: datetime # フォーマットされた日時文字列
    seat_id: int
    menu_id: int
    order_cnt: int
    menu: MenuOut = Field(..., description="メニュー情報")