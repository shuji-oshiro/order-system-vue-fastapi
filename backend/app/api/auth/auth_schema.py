from typing import Optional
from pydantic import Field, field_validator
from backend.app.schemas.baseSchema import BaseSchema


class UserBase(BaseSchema):
    """ユーザーの基本スキーマ"""
    username: str = Field(..., min_length=1, max_length=100, description="ユーザー名")
    password: Optional[str] = Field(None, max_length=500, description="ユーザーのパスワード")

    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if not v or v.strip() == "":
            raise ValueError('ユーザー名は必須です')
        if len(v.strip()) < 1:
            raise ValueError('ユーザー名は1文字以上である必要があります')
        return v.strip()

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if v is not None:
            return v.strip() if v.strip() else None
        return v



class UserOut(UserBase):
    """ユーザー出力スキーマ"""
    id: int = Field(..., description="ユーザーID")
    email: Optional[str] = Field(None, max_length=100, description="メールアドレス")
    class Config:
        from_attributes = True


class userinfoUpdate(BaseSchema):
    """ユーザー情報更新時の入力スキーマ（部分更新対応）"""
    username: Optional[str] = Field(None, min_length=1, max_length=50, description="ユーザー名")
    password: Optional[str] = Field(None, max_length=50, description="ユーザーのパスワード")

    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if v is not None:
            if not v or v.strip() == "":
                raise ValueError('ユーザー名が指定された場合は空にできません')
            if len(v.strip()) < 1:
                raise ValueError('ユーザー名は1文字以上である必要があります')
            return v.strip()
        return v

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if v is not None:
            if not v or v.strip() == "":
                raise ValueError('パスワードは必須です')
            if len(v.strip()) < 6:
                raise ValueError('パスワードは6文字以上である必要があります')
            return v.strip()
        return v