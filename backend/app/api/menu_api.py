# backend/app/api/menu_api.py
import csv
from typing import List, Optional
from collections import defaultdict
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from backend.app.crud import menu_crud
from backend.app.database.database import get_db
from backend.app.schemas.menu_schema import (
    MenuIn, MenuOut, MenuUpdate, MenuUpdateLegacy, MenuBulkCreateRequest, 
    MenuBulkCreateResponse, MenuSearchResult
)

router = APIRouter()


@router.get("/", response_model=List[MenuOut])
def get_menus(
    skip: int = Query(0, ge=0, description="スキップ件数"),
    limit: int = Query(100, ge=1, le=1000, description="取得上限件数"),
    db: Session = Depends(get_db)
):
    """
    全メニュー情報を取得する（ページネーション対応）
    
    Args:
        skip (int): スキップ件数
        limit (int): 取得上限件数
        
    Returns:
        List[MenuOut]: メニュー情報のリスト
        
    Raises:
        HTTPException: データベースエラー時
    """
    menus = menu_crud.get_menus(db, skip=skip, limit=limit)
    return menus


@router.get("/count")
def get_menu_count(db: Session = Depends(get_db)):
    """
    メニューの総数を取得する
    
    Returns:
        dict: メニュー総数
    """
    count = menu_crud.get_menu_count(db)
    return {"count": count}


@router.get("/search", response_model=List[MenuOut])
def search_menus(
    q: str = Query(..., min_length=1, description="検索クエリ"),
    limit: int = Query(20, ge=1, le=100, description="取得上限件数"),
    db: Session = Depends(get_db)
):
    """
    メニューを検索する
    
    Args:
        q (str): 検索クエリ
        limit (int): 取得上限件数
        
    Returns:
        List[MenuOut]: 検索結果のメニューリスト
    """
    menus = menu_crud.search_menus(db, query=q, limit=limit)
    return menus


@router.get("/{menu_id}", response_model=MenuOut)
def get_menu(menu_id: int, db: Session = Depends(get_db)):
    """
    指定されたIDのメニュー情報を取得する
    
    Args:
        menu_id (int): メニューID
        
    Returns:
        MenuOut: メニュー情報
        
    Raises:
        HTTPException: メニューが見つからない場合
    """
    menu = menu_crud.get_menu_by_id(db, menu_id)
    if not menu:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID {menu_id} のメニューが見つかりません"
        )
    return menu


@router.get("/category/{category_id}", response_model=List[MenuOut])
def get_menus_by_category(category_id: int, db: Session = Depends(get_db)):
    """
    指定されたカテゴリのメニュー情報を取得する
    
    Args:
        category_id (int): カテゴリID
        
    Returns:
        List[MenuOut]: メニュー情報のリスト
        
    Raises:
        HTTPException: カテゴリが見つからない場合
    """
    menus = menu_crud.get_menus_by_category(db, category_id)
    return menus


@router.post("/", response_model=MenuOut, status_code=status.HTTP_201_CREATED)
def create_menu(menu: MenuIn, db: Session = Depends(get_db)):
    """
    新しいメニューを作成する
    
    Args:
        menu (MenuIn): 作成するメニュー情報
        
    Returns:
        MenuOut: 作成されたメニュー情報
        
    Raises:
        HTTPException: バリデーションエラーまたはデータベースエラー時
    """
    return menu_crud.create_menu(db, menu)


@router.post("/bulk", response_model=MenuBulkCreateResponse, status_code=status.HTTP_201_CREATED)
def create_multiple_menus(request: MenuBulkCreateRequest, db: Session = Depends(get_db)):
    """
    複数のメニューを一括で作成する
    
    Args:
        request (MenuBulkCreateRequest): 一括作成リクエスト
        
    Returns:
        MenuBulkCreateResponse: 作成結果
        
    Raises:
        HTTPException: バリデーションエラーまたはデータベースエラー時
    """
    created_menus = menu_crud.create_menus_bulk(db, request.menus)
    # 型変換して返す
    menu_outs = [MenuOut.model_validate(menu) for menu in created_menus]
    return MenuBulkCreateResponse(
        created_count=len(created_menus),
        menus=menu_outs
    )


@router.post("/import", response_model=MenuBulkCreateResponse, status_code=status.HTTP_201_CREATED)
def import_menus_from_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    CSVファイルからメニューを一括インポートする
    
    CSV形式: category_id, name, price, description, search_text
    
    Args:
        file (UploadFile): CSVファイル
        
    Returns:
        MenuBulkCreateResponse: インポート結果
        
    Raises:
        HTTPException: ファイル形式エラーまたはデータベースエラー時
    """
    try:
        # ファイル形式チェック
        if not file.filename or not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSVファイルのみサポートしています"
            )
        
        # CSVデータを読み込み
        contents = file.file.read().decode("utf-8")
        reader = csv.reader(contents.splitlines())
        
        # ヘッダーをスキップ（オプション）
        try:
            next(reader)  # ヘッダー行をスキップ
        except StopIteration:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSVファイルが空です"
            )
        
        menus = []
        for row_num, row in enumerate(reader, start=2):  # ヘッダーの次の行から開始
            try:
                if len(row) != 5:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"行 {row_num}: CSVの列数が正しくありません（5列必要）"
                    )
                
                category_id, name, price, description, search_text = row
                
                menu = MenuIn(
                    category_id=int(category_id),
                    name=name.strip(),
                    price=int(price),
                    description=description.strip() if description.strip() else None,
                    search_text=search_text.strip()
                )
                menus.append(menu)
                
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"行 {row_num}: データ形式エラー - {str(e)}"
                )
        
        if not menus:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="インポートするメニューデータがありません"
            )
        
        # 一括作成実行
        created_menus = menu_crud.create_menus_bulk(db, menus)
        
        # 型変換して返す
        menu_outs = [MenuOut.model_validate(menu) for menu in created_menus]
        
        return MenuBulkCreateResponse(
            created_count=len(created_menus),
            menus=menu_outs
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSVファイルの処理中にエラーが発生しました: {str(e)}"
        )


@router.put("/{menu_id}", response_model=MenuOut)
def update_menu(menu_id: int, menu_update: MenuUpdate, db: Session = Depends(get_db)):
    """
    指定されたIDのメニュー情報を更新する
    
    Args:
        menu_id (int): 更新するメニューのID
        menu_update (MenuUpdate): 更新内容
        
    Returns:
        MenuOut: 更新されたメニュー情報
        
    Raises:
        HTTPException: メニューが見つからない場合またはバリデーションエラー時
    """
    return menu_crud.update_menu(db, menu_id, menu_update)


@router.delete("/{menu_id}")
def delete_menu(menu_id: int, db: Session = Depends(get_db)):
    """
    指定されたIDのメニューを削除する
    
    Args:
        menu_id (int): 削除するメニューのID
        
    Returns:
        dict: 削除結果メッセージ
        
    Raises:
        HTTPException: メニューが見つからない場合または依存関係がある場合
    """
    return menu_crud.delete_menu(db, menu_id)


# 従来のAPIエンドポイント（互換性のため残す）
@router.put("/", response_model=List[MenuOut])
def add_menu_legacy(menu: MenuIn, db: Session = Depends(get_db)):
    """
    メニュー情報を追加する（従来互換）
    
    Args:
        menu (MenuIn): 追加するメニュー情報
        
    Returns:
        List[MenuOut]: 全メニュー情報のリスト
        
    Raises:
        HTTPException: バリデーションエラーまたはデータベースエラー時
    """
    return menu_crud.add_menu(db, menu)


@router.patch("/", response_model=List[MenuOut])
def update_menu_legacy(menu_update: MenuUpdateLegacy, db: Session = Depends(get_db)):
    """
    メニュー情報を更新する（従来互換）
    
    Args:
        menu_update (MenuUpdateLegacy): 更新内容（menu_idを含む）
        
    Returns:
        List[MenuOut]: 全メニュー情報のリスト
        
    Raises:
        HTTPException: メニューが見つからない場合またはバリデーションエラー時
    """
    # 新しいMenuUpdateスキーマに変換
    update_data = menu_update.model_dump(exclude={'menu_id'})
    new_menu_update = MenuUpdate(**update_data)
    
    menu_crud.update_menu(db, menu_update.menu_id, new_menu_update)
    return menu_crud.get_menus(db)