# backend/app/crud/menu_crud.py
from typing import List, Optional
from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy import or_
import backend.app.models.model as model
from backend.app.schemas.menu_schema import MenuIn, MenuUpdate


# メニュー情報のCRUD操作を行うモジュール

def get_menus(db: Session, skip: int = 0, limit: int = 100):
    """
    全メニュー情報を取得する（ページネーション対応）
    
    Args:
        db (Session): データベースセッション
        skip (int): スキップ件数
        limit (int): 取得上限件数
        
    Returns:
        List[Menu]: メニュー情報のリスト
        
    Raises:
        HTTPException: データベースエラー時
    """
    try:
        menus = db.query(model.Menu)\
                 .options(joinedload(model.Menu.category))\
                 .offset(skip)\
                 .limit(limit)\
                 .all()
        return menus
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"メニュー情報の取得中にエラーが発生しました: {str(e)}"
        )


def get_menu_by_id(db: Session, menu_id: int):
    """
    指定されたIDのメニュー情報を取得する
    
    Args:
        db (Session): データベースセッション
        menu_id (int): メニューID
        
    Returns:
        Menu: メニュー情報（見つからない場合はNone）
        
    Raises:
        HTTPException: データベースエラー時
    """
    try:
        menu = db.query(model.Menu)\
                .options(joinedload(model.Menu.category))\
                .filter(model.Menu.id == menu_id)\
                .first()
        return menu
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"メニュー情報の取得中にエラーが発生しました: {str(e)}"
        )


def get_menus_by_category(db: Session, category_id: int):
    """
    指定されたカテゴリのメニュー情報を取得する
    
    Args:
        db (Session): データベースセッション
        category_id (int): カテゴリID
        
    Returns:
        List[Menu]: メニュー情報のリスト
        
    Raises:
        HTTPException: データベースエラー時
    """
    try:
        # カテゴリの存在確認
        category = db.query(model.Category).filter(model.Category.id == category_id).first()
        if not category:
            raise HTTPException(
                status_code=404,
                detail=f"ID {category_id} のカテゴリが見つかりません"
            )
        
        menus = db.query(model.Menu)\
                 .options(joinedload(model.Menu.category))\
                 .filter(model.Menu.category_id == category_id)\
                 .all()
        return menus
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"メニュー情報の取得中にエラーが発生しました: {str(e)}"
        )


def search_menus(db: Session, query: str, limit: int = 20):
    """
    メニューを検索する（名前と検索テキストで部分一致）
    
    Args:
        db (Session): データベースセッション
        query (str): 検索クエリ
        limit (int): 取得上限件数
        
    Returns:
        List[Menu]: 検索結果のメニューリスト
        
    Raises:
        HTTPException: データベースエラー時
    """
    try:
        menus = db.query(model.Menu)\
                 .options(joinedload(model.Menu.category))\
                 .filter(
                     or_(
                         model.Menu.name.contains(query),
                         model.Menu.search_text.contains(query)
                     )
                 )\
                 .limit(limit)\
                 .all()
        return menus
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"メニュー検索中にエラーが発生しました: {str(e)}"
        )


def create_menu(db: Session, menu: MenuIn):
    """
    新しいメニューを作成する
    
    Args:
        db (Session): データベースセッション
        menu (MenuIn): 作成するメニュー情報
        
    Returns:
        Menu: 作成されたメニュー情報
        
    Raises:
        HTTPException: バリデーションエラーまたはデータベースエラー時
    """
    try:
        # カテゴリの存在確認
        category = db.query(model.Category).filter(model.Category.id == menu.category_id).first()
        if not category:
            raise HTTPException(
                status_code=400,
                detail=f"ID {menu.category_id} のカテゴリが見つかりません"
            )
        
        # 同名のメニューが既に存在するかチェック
        existing_menu = db.query(model.Menu).filter(model.Menu.name == menu.name).first()
        if existing_menu:
            raise HTTPException(
                status_code=400, 
                detail=f"メニュー名 '{menu.name}' は既に存在します"
            )
        
        # 新しいメニューを作成
        db_menu = model.Menu(**menu.model_dump())
        db.add(db_menu)
        db.commit()
        db.refresh(db_menu)
        
        # カテゴリ情報付きで返す
        return get_menu_by_id(db, db_menu.id)
        
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=400, 
            detail=f"データの整合性エラー: {str(e)}"
        )
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"メニューの作成中にエラーが発生しました: {str(e)}"
        )


def create_menus_bulk(db: Session, menus: List[MenuIn]):
    """
    複数のメニューを一括で作成する
    
    Args:
        db (Session): データベースセッション
        menus (List[MenuIn]): 作成するメニュー情報のリスト
        
    Returns:
        List[Menu]: 作成されたメニュー情報のリスト
        
    Raises:
        HTTPException: バリデーションエラーまたはデータベースエラー時
    """
    try:
        # カテゴリIDの一覧を取得して存在確認
        category_ids = {menu.category_id for menu in menus}
        existing_categories = db.query(model.Category.id).filter(
            model.Category.id.in_(category_ids)
        ).all()
        existing_category_ids = {cat.id for cat in existing_categories}
        
        invalid_category_ids = category_ids - existing_category_ids
        if invalid_category_ids:
            raise HTTPException(
                status_code=400,
                detail=f"以下のカテゴリIDが見つかりません: {', '.join(map(str, invalid_category_ids))}"
            )
        
        # 重複チェック：既存のメニュー名を取得
        existing_names = {menu.name for menu in db.query(model.Menu).all()}
        
        # 新規メニュー名をチェック
        new_names = {menu.name for menu in menus}
        duplicates = existing_names.intersection(new_names)
        
        if duplicates:
            raise HTTPException(
                status_code=400, 
                detail=f"以下のメニュー名は既に存在します: {', '.join(duplicates)}"
            )
        
        # 入力データ内での重複チェック
        if len(new_names) != len(menus):
            raise HTTPException(
                status_code=400, 
                detail="入力データに重複するメニュー名があります"
            )
        
        # 一括作成
        db_menus = [model.Menu(**menu.model_dump()) for menu in menus]
        db.add_all(db_menus)
        db.commit()
        
        # 作成されたメニューを再取得して返す
        created_ids = [menu.id for menu in db_menus]
        return db.query(model.Menu)\
                .options(joinedload(model.Menu.category))\
                .filter(model.Menu.id.in_(created_ids))\
                .all()
        
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=400, 
            detail=f"データの整合性エラー: {str(e)}"
        )
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"メニューの一括作成中にエラーが発生しました: {str(e)}"
        )


def update_menu(db: Session, menu_id: int, menu_update: MenuUpdate):
    """
    指定されたIDのメニュー情報を更新する
    
    Args:
        db (Session): データベースセッション
        menu_id (int): 更新するメニューのID
        menu_update (MenuUpdate): 更新内容
        
    Returns:
        Menu: 更新されたメニュー情報
        
    Raises:
        HTTPException: メニューが見つからない場合またはデータベースエラー時
    """
    try:
        # 更新対象のメニューを取得
        db_menu = db.query(model.Menu).filter(model.Menu.id == menu_id).first()
        
        if not db_menu:
            raise HTTPException(
                status_code=404, 
                detail=f"ID {menu_id} のメニューが見つかりません"
            )
        
        # 更新データを適用（Noneでない値のみ）
        update_data = menu_update.model_dump(exclude_unset=True)
        
        # カテゴリIDが変更される場合、存在確認
        if 'category_id' in update_data:
            category = db.query(model.Category).filter(
                model.Category.id == update_data['category_id']
            ).first()
            if not category:
                raise HTTPException(
                    status_code=400,
                    detail=f"ID {update_data['category_id']} のカテゴリが見つかりません"
                )
        
        # 名前が変更される場合、重複チェック
        if 'name' in update_data and update_data['name'] != db_menu.name:
            existing_menu = db.query(model.Menu).filter(
                model.Menu.name == update_data['name'],
                model.Menu.id != menu_id
            ).first()
            
            if existing_menu:
                raise HTTPException(
                    status_code=400, 
                    detail=f"メニュー名 '{update_data['name']}' は既に存在します"
                )
        
        # データを更新
        for field, value in update_data.items():
            setattr(db_menu, field, value)
        
        db.commit()
        db.refresh(db_menu)
        
        # カテゴリ情報付きで返す
        return get_menu_by_id(db, menu_id)
        
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"メニューの更新中にエラーが発生しました: {str(e)}"
        )


def delete_menu(db: Session, menu_id: int):
    """
    指定されたIDのメニューを削除する
    
    Args:
        db (Session): データベースセッション
        menu_id (int): 削除するメニューのID
        
    Returns:
        dict: 削除結果メッセージ
        
    Raises:
        HTTPException: メニューが見つからない場合、依存関係がある場合、またはデータベースエラー時
    """
    try:
        # 削除対象のメニューを取得
        db_menu = db.query(model.Menu).filter(model.Menu.id == menu_id).first()
        
        if not db_menu:
            raise HTTPException(
                status_code=404, 
                detail=f"ID {menu_id} のメニューが見つかりません"
            )
        
        # 依存関係チェック：このメニューに紐づく注文があるかチェック
        order_count = db.query(model.Order).filter(model.Order.menu_id == menu_id).count()
        
        if order_count > 0:
            raise HTTPException(
                status_code=400, 
                detail=f"このメニューには {order_count} 件の注文が紐づいているため削除できません"
            )
        
        # メニューを削除
        db.delete(db_menu)
        db.commit()
        
        return {"message": f"メニュー '{db_menu.name}' (ID: {menu_id}) を削除しました"}
        
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"メニューの削除中にエラーが発生しました: {str(e)}"
        )


def menu_exists(db: Session, menu_id: int) -> bool:
    """
    指定されたIDのメニューが存在するかチェックする
    
    Args:
        db (Session): データベースセッション
        menu_id (int): チェックするメニューID
        
    Returns:
        bool: メニューが存在する場合True、存在しない場合False
    """
    try:
        count = db.query(model.Menu).filter(model.Menu.id == menu_id).count()
        return count > 0
    except SQLAlchemyError:
        return False


def get_menu_count(db: Session) -> int:
    """
    メニューの総数を取得する
    
    Args:
        db (Session): データベースセッション
        
    Returns:
        int: メニューの総数
    """
    try:
        return db.query(model.Menu).count()
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"メニュー数の取得中にエラーが発生しました: {str(e)}"
        )


def get_all_menus(db: Session):
    """
    全メニュー情報を取得する（ML学習用）
    
    Args:
        db (Session): データベースセッション
        
    Returns:
        List[Menu]: 全メニュー情報のリスト
        
    Raises:
        HTTPException: データベースエラー時
    """
    try:
        menus = db.query(model.Menu)\
                 .options(joinedload(model.Menu.category))\
                 .all()
        return menus
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"全メニュー情報の取得中にエラーが発生しました: {str(e)}"
        )


# 従来のCSVインポート機能（互換性のため残す）
def import_menus_from_csv(db: Session, menus: List[MenuIn]):
    """
    CSVファイルからメニュー情報を一括でインポートする（従来互換）
    
    Args:
        db (Session): データベースセッション
        menus (List[MenuIn]): インポートするメニュー情報のリスト
        
    Returns:
        List[Menu]: インポートされたメニュー情報のリスト
        
    Raises:
        HTTPException: バリデーションエラーまたはデータベースエラー時
    """
    return create_menus_bulk(db, menus)


# 従来のadd_menu関数（互換性のため残す）  
def add_menu(db: Session, menu: MenuIn):
    """
    メニュー情報を追加する（従来互換）
    
    Args:
        db (Session): データベースセッション
        menu (MenuIn): 追加するメニュー情報
        
    Returns:
        List[Menu]: 全メニュー情報のリスト
        
    Raises:
        HTTPException: バリデーションエラーまたはデータベースエラー時
    """
    create_menu(db, menu)
    return get_menus(db)
