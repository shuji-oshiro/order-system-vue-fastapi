# app/main.py
from collections import defaultdict
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends,HTTPException
from backend.app.crud import menu_crud
from backend.app.database.database import get_db
from backend.app.schemas.menu_schema import MenuOut_SP


router = APIRouter()

# 全てのメニュー情報を取得
# 取得したいメニュー情報は、全てのメニュー情報を取得する
@router.get("/", response_model=list[MenuOut_SP])
def get_all_menus_for_category(db: Session = Depends(get_db)):

    try:
        # メニュー情報を取得
        result = menu_crud.get_menus(db)
        if len(result) == 0:
            raise HTTPException(status_code=404, detail="No menus found")

        grouped = defaultdict(list)

        # メニュー情報をカテゴリごとにグループ化
        # defaultdictを使用して、カテゴリIDとカテゴリ名をキーにしてメニューをグループ化
        for menu in result:
            grouped[(menu.category.id, menu.category.name)].append(menu)

        # list[MenuOut_SP]型に変換
        menu_out_list = []
        for (category_id, category_name), menus in grouped.items():
            menu_out_list.append(
                MenuOut_SP(
                    category_id=category_id,
                    category_name=category_name,
                    menues=menus
                )
            )
        return menu_out_list
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'{e}')