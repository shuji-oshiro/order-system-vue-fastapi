from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.app.schemas.menu_schema import MenuOut
from backend.app.database.database import get_db
from backend.app.service.recommend_menu import recommend_menu_by_menu_id

router = APIRouter()

@router.get("/{menu_id}", response_model=MenuOut)
def recommend(menu_id: int, db: Session = Depends(get_db)):
    """
    メニューIDに基づいておすすめメニューを取得する
    
    Args:
        menu_id (int): メニューID
        db (Session): データベースセッション
        
    Returns:
        MenuOut: おすすめメニュー情報
    """
    try:
        menu = recommend_menu_by_menu_id(menu_id, db)
        return menu
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
