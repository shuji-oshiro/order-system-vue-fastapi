# backend/app/crud/category_crud.py
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import backend.app.models.model as model
from backend.app.schemas.category_schema import CategoryIn


# カテゴリ情報のCRUD操作を行うモジュール

def get_categories(db: Session):
    """
    全カテゴリ情報を取得する
    
    Args:
        db (Session): データベースセッション
        
    Returns:
        list[Category]: カテゴリ情報のリスト
        
    Raises:
        HTTPException: データベースエラー時
    """
    try:
        categories = db.query(model.Category).all()
        return categories
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"カテゴリ情報の取得中にエラーが発生しました: {str(e)}"
        )


def get_category_by_id(db: Session, category_id: int):
    """
    指定されたIDのカテゴリ情報を取得する
    
    Args:
        db (Session): データベースセッション
        category_id (int): カテゴリID
        
    Returns:
        Category: カテゴリ情報（見つからない場合はNone）
        
    Raises:
        HTTPException: データベースエラー時
    """
    try:
        category = db.query(model.Category).filter(model.Category.id == category_id).first()
        return category
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"カテゴリ情報の取得中にエラーが発生しました: {str(e)}"
        )


def create_category(db: Session, category: CategoryIn):
    """
    新しいカテゴリを作成する
    
    Args:
        db (Session): データベースセッション
        category (CategoryIn): 作成するカテゴリ情報
        
    Returns:
        Category: 作成されたカテゴリ情報
        
    Raises:
        HTTPException: バリデーションエラーまたはデータベースエラー時
    """
    try:
        # 同名のカテゴリが既に存在するかチェック
        existing_category = db.query(model.Category).filter(
            model.Category.name == category.name
        ).first()
        
        if existing_category:
            raise HTTPException(
                status_code=400, 
                detail=f"カテゴリ名 '{category.name}' は既に存在します"
            )
        
        # 新しいカテゴリを作成
        db_category = model.Category(**category.model_dump())
        db.add(db_category)
        db.commit()
        db.refresh(db_category)
        
        return db_category
        
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
            detail=f"カテゴリの作成中にエラーが発生しました: {str(e)}"
        )


def create_categories_bulk(db: Session, categories: list[CategoryIn]):
    """
    複数のカテゴリを一括で作成する
    
    Args:
        db (Session): データベースセッション
        categories (list[CategoryIn]): 作成するカテゴリ情報のリスト
        
    Returns:
        list[Category]: 作成されたカテゴリ情報のリスト
        
    Raises:
        HTTPException: バリデーションエラーまたはデータベースエラー時
    """
    try:
        # 重複チェック：既存のカテゴリ名を取得
        existing_names = {cat.name for cat in db.query(model.Category).all()}
        
        # 新規カテゴリ名をチェック
        new_names = {cat.name for cat in categories}
        duplicates = existing_names.intersection(new_names)
        
        if duplicates:
            raise HTTPException(
                status_code=400, 
                detail=f"以下のカテゴリ名は既に存在します: {', '.join(duplicates)}"
            )
        
        # 入力データ内での重複チェック
        if len(new_names) != len(categories):
            raise HTTPException(
                status_code=400, 
                detail="入力データに重複するカテゴリ名があります"
            )
        
        # 一括作成
        db_categories = [model.Category(**cat.model_dump()) for cat in categories]
        db.add_all(db_categories)
        db.commit()
        
        # 作成されたカテゴリを再取得して返す
        for db_cat in db_categories:
            db.refresh(db_cat)
        
        return db_categories
        
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
            detail=f"カテゴリの一括作成中にエラーが発生しました: {str(e)}"
        )


def update_category(db: Session, category_id: int, category_update: CategoryIn):
    """
    指定されたIDのカテゴリ情報を更新する
    
    Args:
        db (Session): データベースセッション
        category_id (int): 更新するカテゴリのID
        category_update (CategoryIn): 更新内容
        
    Returns:
        Category: 更新されたカテゴリ情報
        
    Raises:
        HTTPException: カテゴリが見つからない場合またはデータベースエラー時
    """
    try:
        # 更新対象のカテゴリを取得
        db_category = db.query(model.Category).filter(model.Category.id == category_id).first()
        
        if not db_category:
            raise HTTPException(
                status_code=404, 
                detail=f"ID {category_id} のカテゴリが見つかりません"
            )
        
        # 名前が変更される場合、重複チェック
        if category_update.name != db_category.name:
            existing_category = db.query(model.Category).filter(
                model.Category.name == category_update.name,
                model.Category.id != category_id
            ).first()
            
            if existing_category:
                raise HTTPException(
                    status_code=400, 
                    detail=f"カテゴリ名 '{category_update.name}' は既に存在します"
                )
        
        # データを更新
        for field, value in category_update.model_dump().items():
            setattr(db_category, field, value)
        
        db.commit()
        db.refresh(db_category)
        
        return db_category
        
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"カテゴリの更新中にエラーが発生しました: {str(e)}"
        )


def delete_category(db: Session, category_id: int):
    """
    指定されたIDのカテゴリを削除する
    
    Args:
        db (Session): データベースセッション
        category_id (int): 削除するカテゴリのID
        
    Returns:
        dict: 削除結果メッセージ
        
    Raises:
        HTTPException: カテゴリが見つからない場合、依存関係がある場合、またはデータベースエラー時
    """
    try:
        # 削除対象のカテゴリを取得
        db_category = db.query(model.Category).filter(model.Category.id == category_id).first()
        
        if not db_category:
            raise HTTPException(
                status_code=404, 
                detail=f"ID {category_id} のカテゴリが見つかりません"
            )
        
        # 依存関係チェック：このカテゴリに紐づくメニューがあるかチェック
        menu_count = db.query(model.Menu).filter(model.Menu.category_id == category_id).count()
        
        if menu_count > 0:
            raise HTTPException(
                status_code=400, 
                detail=f"このカテゴリには {menu_count} 件のメニューが紐づいているため削除できません"
            )
        
        # カテゴリを削除
        db.delete(db_category)
        db.commit()
        
        return {"message": f"カテゴリ '{db_category.name}' (ID: {category_id}) を削除しました"}
        
    except HTTPException:
        db.rollback()
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"カテゴリの削除中にエラーが発生しました: {str(e)}"
        )


def category_exists(db: Session, category_id: int) -> bool:
    """
    指定されたIDのカテゴリが存在するかチェックする
    
    Args:
        db (Session): データベースセッション
        category_id (int): チェックするカテゴリID
        
    Returns:
        bool: カテゴリが存在する場合True、存在しない場合False
    """
    try:
        count = db.query(model.Category).filter(model.Category.id == category_id).count()
        return count > 0
    except SQLAlchemyError:
        return False