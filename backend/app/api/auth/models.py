from datetime import datetime
from sqlalchemy import String,func,ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.database.database import Base

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True, index=True) # 主キー
    username: Mapped[str] = mapped_column(String, index=True, unique=True) # ユーザー名
    email: Mapped[str] = mapped_column(String, index=True, unique=True,nullable=True) # メールアドレス
    password: Mapped[str] = mapped_column(String) # パスワード（ハッシュ化されたもの）
    is_active: Mapped[bool] = mapped_column(default=True) # アクティブフラグ