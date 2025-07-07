from typing import Optional, List
from pydantic import Field, field_validator
from decimal import Decimal
from .baseSchema import BaseSchema


class CategoryBase(BaseSchema):
    """カテゴリ基本情報"""
    name: str = Field(..., description="カテゴリ名")
    description: Optional[str] = Field(None, description="カテゴリの説明")


class MenuBase(BaseSchema):
    """メニューの基本スキーマ"""
    name: str = Field(..., min_length=1, max_length=100, description="メニュー名")
    price: int = Field(..., ge=0, le=999999, description="価格（円）")
    description: Optional[str] = Field(None, max_length=500, description="メニューの説明")
    search_text: str = Field(..., min_length=1, max_length=200, description="検索テキスト")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or v.strip() == "":
            raise ValueError('メニュー名は必須です')
        return v.strip()
    
    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v < 0:
            raise ValueError('価格は0円以上である必要があります')
        if v > 999999:
            raise ValueError('価格は999,999円以下である必要があります')
        return v
    
    @field_validator('search_text')
    @classmethod
    def validate_search_text(cls, v):
        if not v or v.strip() == "":
            raise ValueError('検索テキストは必須です')
        return v.strip()
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if v is not None:
            return v.strip() if v.strip() else None
        return v


class MenuIn(MenuBase):
    """メニュー作成時の入力スキーマ"""
    category_id: int = Field(..., gt=0, description="カテゴリID")
    
    @field_validator('category_id')
    @classmethod
    def validate_category_id(cls, v):
        if v <= 0:
            raise ValueError('カテゴリIDは1以上である必要があります')
        return v


class MenuOut(MenuBase):
    """メニュー出力スキーマ"""
    id: int = Field(..., description="メニューID")
    category_id: int = Field(..., description="カテゴリID")
    category: CategoryBase = Field(..., description="カテゴリ情報")
    
    class Config:
        from_attributes = True


class MenuUpdate(BaseSchema):
    """メニュー更新時の入力スキーマ（部分更新対応）"""
    category_id: Optional[int] = Field(None, gt=0, description="カテゴリID")
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="メニュー名")
    price: Optional[int] = Field(None, ge=0, le=999999, description="価格（円）")
    description: Optional[str] = Field(None, max_length=500, description="メニューの説明")
    search_text: Optional[str] = Field(None, min_length=1, max_length=200, description="検索テキスト")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if v is not None:
            if not v or v.strip() == "":
                raise ValueError('メニュー名が指定された場合は空にできません')
            return v.strip()
        return v
    
    @field_validator('price')
    @classmethod
    def validate_price(cls, v):
        if v is not None:
            if v < 0:
                raise ValueError('価格は0円以上である必要があります')
            if v > 999999:
                raise ValueError('価格は999,999円以下である必要があります')
        return v
    
    @field_validator('search_text')
    @classmethod
    def validate_search_text(cls, v):
        if v is not None:
            if not v or v.strip() == "":
                raise ValueError('検索テキストが指定された場合は空にできません')
            return v.strip()
        return v
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if v is not None:
            return v.strip() if v.strip() else None
        return v


class MenuUpdateLegacy(BaseSchema):
    """従来互換のメニュー更新スキーマ（menu_id付き）"""
    menu_id: int = Field(..., gt=0, description="メニューID")
    category_id: int = Field(..., gt=0, description="カテゴリID")
    name: str = Field(..., min_length=1, max_length=100, description="メニュー名")
    price: int = Field(..., ge=0, le=999999, description="価格（円）")
    description: Optional[str] = Field(None, max_length=500, description="メニューの説明")
    search_text: str = Field(..., min_length=1, max_length=200, description="検索テキスト")


class MenuOut_SP(BaseSchema):
    """カテゴリ別メニューリスト出力スキーマ"""
    category_id: int = Field(..., description="カテゴリID")
    category_name: str = Field(..., description="カテゴリ名")
    menus: List[MenuOut] = Field(..., description="メニューリスト")  # typo修正: menues -> menus
    
    class Config:
        from_attributes = True


class MenuSearchResult(BaseSchema):
    """メニュー検索結果スキーマ"""
    menu: MenuOut = Field(..., description="メニュー情報")
    score: float = Field(..., ge=0.0, le=1.0, description="検索スコア")
    
    class Config:
        from_attributes = True


class MenuBulkCreateRequest(BaseSchema):
    """メニュー一括作成リクエストスキーマ"""
    menus: List[MenuIn] = Field(..., description="作成するメニューリスト")
    
    @field_validator('menus')
    @classmethod
    def validate_menus_not_empty(cls, v):
        if not v or len(v) == 0:
            raise ValueError('メニューリストは空にできません')
        if len(v) > 100:  # 一括作成の上限設定
            raise ValueError('一度に作成できるメニューは100件までです')
        return v


class MenuBulkCreateResponse(BaseSchema):
    """メニュー一括作成レスポンススキーマ"""
    created_count: int = Field(..., description="作成されたメニュー数")
    menus: List[MenuOut] = Field(..., description="作成されたメニューリスト")
    
    class Config:
        from_attributes = True


