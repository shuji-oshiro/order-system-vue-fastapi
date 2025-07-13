# app/crud.py
from fastapi import HTTPException
import backend.app.models.model as model
from sqlalchemy.orm import Session
from backend.app.schemas.order_schema import OrderOut, OrderIn


# 注文情報のCRUD操作を行うモジュール
# 注文情報の取得
# 取得したい注文情報は、座席IDを指定して取得する
def get_orders(db: Session, seat_id: int):
    return db.query(model.Order).filter(model.Order.seat_id == seat_id)


# 特定メニューIDが注文された座席IDリストを取得
def get_seats_by_menu_id(db: Session, menu_id: int):
    """特定のメニューIDが注文された座席IDのリストを取得する"""
    return db.query(model.Order.seat_id).filter(model.Order.menu_id == menu_id).distinct().all()


# 特定の座席IDリストで注文されたメニューIDリストを取得
def get_menu_ids_by_seats(db: Session, seat_ids: list[int]):
    """特定の座席IDリストで注文されたメニューIDのリストを取得する"""
    return db.query(model.Order.menu_id).filter(model.Order.seat_id.in_(seat_ids)).distinct().all()


# 注文情報の追加
# 注文情報を追加する際は、座席ID、メニューID、注文数を指定して追加する
def add_order(db: Session, orders: list[OrderIn]):
    db_orders = []
    for order in orders:
        db_order = model.Order(**order.model_dump())
        db.add(db_order)
        db_orders.append(db_order)

    db.flush() # ここで自動的に ID が入る
    db.commit()     

    return get_orders(db, seat_id=db_orders[0].seat_id)  # 最初の注文の座席IDを返す
    

# 注文情報の削除
# 注文情報を削除する際は、注文IDを指定して削除する
def delete_order(db: Session, order_id: int):
    db_order = db.query(model.Order).filter(model.Order.id == order_id).first()
    if not db_order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    db.delete(db_order)
    db.commit()
    return get_orders(db, seat_id=db_order.seat_id)  # 削除後の注文情報を返す
