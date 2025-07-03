# app/main.py

from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException
from backend.app.crud import order_crud
from backend.app.database.database import get_db
from backend.app.schemas.order_schema import OrderIn,OrderOut


router = APIRouter()

# シート単位の注文情報取得
@router.get("/{seat_id}", response_model=list[OrderOut])
def get_orders(seat_id: int, db: Session = Depends(get_db)):
    try:
        result = list(order_crud.get_orders(db, seat_id))
        if len(result) == 0:
            raise HTTPException(status_code=404, detail="No orders found for this seat")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'{e}')

# 注文情報追加
@router.post("/", response_model=list[OrderOut])
def add_order(orders: list[OrderIn], db: Session = Depends(get_db)):
    try:
        return order_crud.add_order(db, orders)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'{e}')

# 注文情報削除
@router.delete("/{order_id}", response_model=list[OrderOut])
def delete_order(order_id: int, db: Session = Depends(get_db)):
    try:
        return order_crud.delete_order(db, order_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'{e}') 
