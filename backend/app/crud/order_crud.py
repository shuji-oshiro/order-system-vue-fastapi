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


# 頻度ベースレコメンド用: メニューペアの共起頻度を取得
def get_menu_cooccurrence_frequency(db: Session, menu_id: int):
    """
    指定メニューIDと一緒に注文されたメニューの頻度を取得する
    同じ座席で注文されたメニューの組み合わせを分析
    
    Returns:
        List[Tuple]: (menu_id, frequency) のリスト
    """
    from sqlalchemy import func
    
    # サブクエリ: 指定メニューIDが注文された座席を取得
    target_seats = db.query(
        model.Order.seat_id
    ).filter(model.Order.menu_id == menu_id).distinct().subquery()
    
    # メインクエリ: 同じ座席の他のメニューの頻度を計算
    result = db.query(
        model.Order.menu_id,
        func.count(model.Order.menu_id).label('frequency')
    ).join(
        target_seats,
        model.Order.seat_id == target_seats.c.seat_id
    ).filter(
        model.Order.menu_id != menu_id  # 指定メニュー以外
    ).group_by(
        model.Order.menu_id
    ).order_by(
        func.count(model.Order.menu_id).desc()
    ).all()
    
    return result


# 時間帯ベースレコメンド用: 時間帯別人気メニューを取得
def get_popular_menus_by_time_period(db: Session, hour: int, limit: int = 10):
    """
    指定時間帯で人気のメニューを取得する
    
    Args:
        hour: 時間（0-23）
        limit: 取得件数
    """
    from sqlalchemy import func, extract
    
    result = db.query(
        model.Order.menu_id,
        func.count(model.Order.menu_id).label('frequency')
    ).filter(
        extract('hour', model.Order.order_date) == hour
    ).group_by(
        model.Order.menu_id
    ).order_by(
        func.count(model.Order.menu_id).desc()
    ).limit(limit).all()
    
    return result


# カテゴリ親和性レコメンド用: カテゴリ間の共起頻度を取得
def get_category_cooccurrence(db: Session, category_id: int):
    """
    指定カテゴリと一緒に注文される他のカテゴリの頻度を取得
    """
    from sqlalchemy import func
    
    # サブクエリ: 指定カテゴリのメニューが注文された座席を取得
    target_seats = db.query(
        model.Order.seat_id
    ).join(model.Menu).filter(
        model.Menu.category_id == category_id
    ).distinct().subquery()
    
    # 同じ座席の他のカテゴリの頻度
    result = db.query(
        model.Menu.category_id,
        func.count(model.Menu.category_id).label('frequency')
    ).join(model.Order).join(
        target_seats,
        model.Order.seat_id == target_seats.c.seat_id
    ).filter(
        model.Menu.category_id != category_id
    ).group_by(
        model.Menu.category_id
    ).order_by(
        func.count(model.Menu.category_id).desc()
    ).all()
    
    return result


# 価格帯考慮レコメンド用: 価格帯別メニューを取得
def get_menus_by_price_range(db: Session, min_price: int, max_price: int, exclude_menu_id: int | None = None):
    """
    指定価格帯のメニューを取得する
    """
    query = db.query(model.Menu).filter(
        model.Menu.price >= min_price,
        model.Menu.price <= max_price
    )
    
    if exclude_menu_id:
        query = query.filter(model.Menu.id != exclude_menu_id)
    
    return query.all()


# 最近のトレンドレコメンド用: 期間指定での人気メニューを取得
def get_recent_popular_menus(db: Session, days: int = 30, limit: int = 10):
    """
    指定期間内の人気メニューを取得する
    
    Args:
        days: 過去何日間か
        limit: 取得件数
    """
    from sqlalchemy import func
    from datetime import datetime, timedelta
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    result = db.query(
        model.Order.menu_id,
        func.count(model.Order.menu_id).label('frequency'),
        func.max(model.Order.order_date).label('latest_order')
    ).filter(
        model.Order.order_date >= cutoff_date
    ).group_by(
        model.Order.menu_id
    ).order_by(
        func.count(model.Order.menu_id).desc(),
        func.max(model.Order.order_date).desc()
    ).limit(limit).all()
    
    return result


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
