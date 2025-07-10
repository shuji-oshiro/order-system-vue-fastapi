from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from backend.app.api.auth.schemas import TokenResponse, UserCreate, UserCreateResponse
from backend.app.api.auth.utils import create_access_token, verify_password
from backend.app.api.auth.user_crud import get_user_password_by_username, create_user
from backend.app.database.database import get_db
from backend.app.api.auth.models import User
from backend.app.api.auth.utils import get_password_hash
from backend.app.utils.logging_config import get_logger

router = APIRouter()
logger = get_logger("auth")

@router.post("/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    logger.info(f"ログイン試行: {form_data.username}")
    
    user = get_user_password_by_username(db, form_data.username)
    if not user or not verify_password(form_data.password, user.password):
        logger.warning(f"ログイン失敗: {form_data.username}")
        raise HTTPException(status_code=400, detail="ユーザー名またはパスワードが不正です")
    
    token = create_access_token({"sub": user.username})
    logger.info(f"ログイン成功: {form_data.username}")
    return {"access_token": token, "token_type": "bearer"}

@router.post("/create_user", response_model=UserCreateResponse)
def create_new_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    新しいユーザーを作成する
    
    Args:
        user_data (UserCreate): ユーザー作成データ
        db (Session): データベースセッション
        
    Returns:
        UserCreateResponse: 作成結果
    """
    try:
        # 既存のユーザー名をチェック
        existing_user = get_user_password_by_username(db, user_data.username)
        if existing_user:
            return UserCreateResponse(
                success=False,
                message=f"ユーザー名 '{user_data.username}' は既に存在しています"
            )
        
        # 新しいユーザーを作成
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            password=get_password_hash(user_data.password),  # パスワードをハッシュ化
            is_active=True
        )
        
        # データベースに保存
        created_user = create_user(db, new_user)
        
        return UserCreateResponse(
            success=True,
            message=f"ユーザー '{created_user.username}' を正常に作成しました",
            user_id=created_user.id,
            username=created_user.username
        )
        
    except IntegrityError as e:
        db.rollback()
        return UserCreateResponse(
            success=False,
            message="ユーザー名またはメールアドレスが既に存在しています"
        )
    except Exception as e:
        db.rollback()
        return UserCreateResponse(
            success=False,
            message=f"ユーザー作成中にエラーが発生しました: {str(e)}"
        )
