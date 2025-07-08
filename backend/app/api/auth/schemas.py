from pydantic import BaseModel, Field
from typing import Optional

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=100, description="ユーザー名")
    password: str = Field(..., min_length=6, max_length=100, description="パスワード")
    email: Optional[str] = Field(None, max_length=255, description="メールアドレス")

class UserCreateResponse(BaseModel):
    success: bool = Field(..., description="作成成功フラグ")
    message: str = Field(..., description="作成結果メッセージ")
    user_id: Optional[int] = Field(default=None, description="作成されたユーザーID")
    username: Optional[str] = Field(default=None, description="作成されたユーザー名")
