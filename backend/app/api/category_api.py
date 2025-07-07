# backend/app/api/category_api.py
from typing import List
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.crud import category_crud
from backend.app.database.database import get_db
from backend.app.schemas.category_schema import CategoryOut, CategoryIn

router = APIRouter()


@router.get("/", response_model=List[CategoryOut])
def get_categories(db: Session = Depends(get_db)):
    """
    全カテゴリ情報を取得する
    
    Returns:
        List[CategoryOut]: カテゴリ情報のリスト
        
    Raises:
        HTTPException: データベースエラー時
    """
    return category_crud.get_categories(db)


@router.get("/{category_id}", response_model=CategoryOut)
def get_category(category_id: int, db: Session = Depends(get_db)):
    """
    指定されたIDのカテゴリ情報を取得する
    
    Args:
        category_id (int): カテゴリID
        
    Returns:
        CategoryOut: カテゴリ情報
        
    Raises:
        HTTPException: カテゴリが見つからない場合
    """
    category = category_crud.get_category_by_id(db, category_id)
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID {category_id} のカテゴリが見つかりません"
        )
    return category


@router.post("/", response_model=CategoryOut, status_code=status.HTTP_201_CREATED)
def create_single_category(category: CategoryIn, db: Session = Depends(get_db)):
    """
    新しいカテゴリを作成する
    
    Args:
        category (CategoryIn): 作成するカテゴリ情報
        
    Returns:
        CategoryOut: 作成されたカテゴリ情報
        
    Raises:
        HTTPException: バリデーションエラーまたはデータベースエラー時
    """
    return category_crud.create_category(db, category)


@router.post("/bulk", response_model=List[CategoryOut], status_code=status.HTTP_201_CREATED)
def create_multiple_categories(categories: List[CategoryIn], db: Session = Depends(get_db)):
    """
    複数のカテゴリを一括で作成する
    
    Args:
        categories (List[CategoryIn]): 作成するカテゴリ情報のリスト
        
    Returns:
        List[CategoryOut]: 作成されたカテゴリ情報のリスト
        
    Raises:
        HTTPException: バリデーションエラーまたはデータベースエラー時
    """
    if not categories:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="カテゴリデータが空です"
        )
    
    return category_crud.create_categories_bulk(db, categories)


@router.put("/{category_id}", response_model=CategoryOut)
def update_category(category_id: int, category_update: CategoryIn, db: Session = Depends(get_db)):
    """
    指定されたIDのカテゴリ情報を更新する
    
    Args:
        category_id (int): 更新するカテゴリのID
        category_update (CategoryIn): 更新内容
        
    Returns:
        CategoryOut: 更新されたカテゴリ情報
        
    Raises:
        HTTPException: カテゴリが見つからない場合またはバリデーションエラー時
    """
    return category_crud.update_category(db, category_id, category_update)


@router.delete("/{category_id}")
def delete_category(category_id: int, db: Session = Depends(get_db)):
    """
    指定されたIDのカテゴリを削除する
    
    Args:
        category_id (int): 削除するカテゴリのID
        
    Returns:
        dict: 削除結果メッセージ
        
    Raises:
        HTTPException: カテゴリが見つからない場合または依存関係がある場合
    """
    return category_crud.delete_category(db, category_id)

