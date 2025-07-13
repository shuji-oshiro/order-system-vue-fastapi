from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from backend.app.schemas.menu_schema import MenuOut
from backend.app.database.database import get_db
from backend.app.service.recommend_menu import recommend_menu_by_menu_id

router = APIRouter()

@router.get("/{menu_id}", response_model=MenuOut)
def recommend(
    menu_id: int, 
    phase: int = Query(default=1, ge=1, le=4, description="レコメンドフェーズ（1-4）"),
    db: Session = Depends(get_db)
):
    """
    メニューIDに基づいておすすめメニューを取得する
    
    Args:
        menu_id (int): メニューID
        phase (int): レコメンドフェーズ（1: 頻度ベース, 2: 時間帯+カテゴリ, 3: 価格考慮, 4: 複合スコア）
        db (Session): データベースセッション
        
    Returns:
        MenuOut: おすすめメニュー情報
    """
    try:
        menu = recommend_menu_by_menu_id(menu_id, db, phase)
        return menu
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
