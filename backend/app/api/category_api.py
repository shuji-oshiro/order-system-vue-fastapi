# app/main.py

from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException
from backend.app.crud import category_crud
from backend.app.database.database import get_db
from backend.app.schemas.category_schema import CategoryOut, CategoryIn

router = APIRouter()

# カテゴリ情報の取得
@router.get("/", response_model=list[CategoryOut])
def get_categories(db: Session = Depends(get_db)):
    try:
        return category_crud.get_category(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'{e}')

# カテゴリ情報の登録
@router.post("/", response_model=list[CategoryOut])
def create_category(category: list[CategoryIn], db: Session = Depends(get_db)):
    try:
        category_crud.insert_category(db, category)
        return category_crud.get_category(db)  # すべてのカテゴリ情報を返す
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'{e}')

