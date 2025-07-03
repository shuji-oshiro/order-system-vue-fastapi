from fastapi import HTTPException
from sqlalchemy.orm import Session
import backend.app.models.model as model
from backend.app.schemas.menu_schema import MenuIn, MenuUpdate

# メニュー情報のCRUD操作を行うモジュール

# メニュー情報の取得
# 取得したいメニュー情報は、全てのメニュー情報を取得する
def get_menus(db: Session): 
    return db.query(model.Menu).all()

# メニュー情報を取得する関数
# メニューIDを指定してメニュー情報を取得する
def get_menu_by_id(db: Session, menu_id: int):
    return db.query(model.Menu).filter(model.Menu.id == menu_id).all()


def get_menus_by_category(db: Session, category_id: int):
    return db.query(model.Menu).filter(model.Menu.category_id == category_id).all()


#　一括でメニュー情報をCSVファイルからデータベースに挿入する関数
# CSVファイルのフォーマットは、1行目にヘッダーがあり、2行目以降にメニュー情報があると仮定します。
# ヘッダーは、food_name, unit_price, descrption, search_text の順である必要があります。
# メニュー情報をCSVファイルからデータベースに一括で挿入する関数
def import_menus_from_csv(db: Session, menus: list[MenuIn]):
    for menu in menus:            
        db_menu = model.Menu(**menu.model_dump())
        db.add(db_menu)
    
    db.flush()
    db.commit()    
    return get_menus(db)    

# メニュー情報を追加する関数
# メニュー情報を追加する際は、メニュー名、価格、説明、検索テキストを指定して追加する
def add_menu(db: Session, menu: MenuIn):   
    # ユーザーをデータベースに追加
    db_menu = model.Menu(**menu.model_dump())
    db.add(db_menu)
    db.commit()
    db.refresh(db_menu)  # ここで自動的に ID が入る

    return get_menus(db)  
 

# メニュー情報を更新する関数
# メニュー情報を更新する際は、メニューID、メニュー名、価格、説明、検索テキストを指定して更新する
def update_menu(db: Session, menu_update: MenuUpdate):
    # 更新処理
    db_menu = db.query(model.Menu).filter(model.Menu.id == menu_update.menu_id).first()
    if not db_menu:
        raise HTTPException(status_code=404, detail="Menu not found")
      
    db_menu.name = menu_update.name
    db_menu.price = menu_update.price
    db_menu.description = menu_update.description
    db_menu.search_text = menu_update.search_text
    db.commit()
    db.refresh(db_menu)

    return get_menus(db) 

# メニュー情報を削除する関数
# メニュー情報を削除する際は、メニューIDを指定して削除する
def delete_menu(db: Session, menu_id: int):
    db_menu = db.query(model.Menu).filter(model.Menu.id == menu_id).first()
    if not db_menu:
        raise HTTPException(status_code=404, detail="Menu not found")
    
    db.delete(db_menu)
    db.commit()
    
    return get_menus(db) 
