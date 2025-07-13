import random
from sqlalchemy.orm import Session
from backend.app.models.model import Menu
from backend.app.schemas.menu_schema import MenuOut
from backend.app.crud import order_crud


def recommend_menu_by_menu_id(menu_id: int, db: Session) -> MenuOut:
    """
    メニューIDに基づいておすすめメニューを取得する
    座席IDでグループ化し、指定メニューが注文された座席の他のメニューから推薦
    
    Args:
        menu_id (int): 基準となるメニューID
        db (Session): データベースセッション
        
    Returns:
        MenuOut: おすすめメニュー
        
    Raises:
        ValueError: 該当する注文履歴やメニューが存在しない場合
    """
    # 1. 指定メニューIDが注文された座席IDリストを取得
    seat_results = order_crud.get_seats_by_menu_id(db, menu_id)
    
    if not seat_results:
        raise ValueError(f"メニューID {menu_id} の注文履歴が存在しません")
    
    seat_ids = [result.seat_id for result in seat_results]
    
    # 2. それらの座席で注文されたメニューIDリストを取得
    menu_results = order_crud.get_menu_ids_by_seats(db, seat_ids)
    
    if not menu_results:
        raise ValueError("関連する注文履歴が存在しません")
    
    # 3. 指定されたメニューID以外から推薦メニューを選択
    related_menu_ids = [result.menu_id for result in menu_results if result.menu_id != menu_id]
    
    if not related_menu_ids:
        raise ValueError("推薦できるメニューが存在しません")
    
    # 4. ランダムに推薦メニューを選択
    recommended_menu_id = random.choice(related_menu_ids)
    
    # 5. メニュー情報を取得（カテゴリ情報も含む）
    menu = db.query(Menu).filter(Menu.id == recommended_menu_id).first()
    
    if not menu:
        raise ValueError(f"メニューID {recommended_menu_id} が見つかりません")
    
    return MenuOut.model_validate(menu)


