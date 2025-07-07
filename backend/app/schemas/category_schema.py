from typing import Optional
from pydantic import Field, field_validator
from .baseSchema import BaseSchema


class CategoryBase(BaseSchema):
    """カテゴリの基本スキーマ"""
    name: str = Field(..., min_length=1, max_length=100, description="カテゴリ名")
    description: Optional[str] = Field(None, max_length=500, description="カテゴリの説明")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v or v.strip() == "":
            raise ValueError('カテゴリ名は必須です')
        if len(v.strip()) < 1:
            raise ValueError('カテゴリ名は1文字以上である必要があります')
        return v.strip()
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if v is not None:
            return v.strip() if v.strip() else None
        return v


class CategoryIn(CategoryBase):
    """カテゴリ作成・更新時の入力スキーマ"""
    pass


class CategoryOut(CategoryBase):
    """カテゴリ出力スキーマ"""
    id: int = Field(..., description="カテゴリID")
    
    class Config:
        from_attributes = True


class CategoryUpdate(BaseSchema):
    """カテゴリ更新時の入力スキーマ（部分更新対応）"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="カテゴリ名")
    description: Optional[str] = Field(None, max_length=500, description="カテゴリの説明")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if v is not None:
            if not v or v.strip() == "":
                raise ValueError('カテゴリ名が指定された場合は空にできません')
            if len(v.strip()) < 1:
                raise ValueError('カテゴリ名は1文字以上である必要があります')
            return v.strip()
        return v
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if v is not None:
            return v.strip() if v.strip() else None
        return v