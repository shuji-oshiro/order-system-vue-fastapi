from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import SQLAlchemyError
import backend.app.api.auth.models as model
from fastapi.security import OAuth2PasswordRequestForm

def get_user_password_by_username(db: Session, username: str):
    """
    指定されたユーザー名のパスワード情報を取得する
    
    Args:
        db (Session): データベースセッション
        username (str): ユーザー名

    Returns:
        User: ユーザー情報（見つからない場合はNone）

    Raises:
        HTTPException: データベースエラー時
    """
    try:
        user = db.query(model.User)\
                .filter(model.User.username == username)\
                .first()
        return user
    except SQLAlchemyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"ユーザー情報の取得中にエラーが発生しました: {str(e)}"
        )
    

def create_user(db: Session, user: model.User):
    """
    新しいユーザーを作成する
    
    Args:
        db (Session): データベースセッション
        user (User): 作成するユーザー情報

    Returns:
        User: 作成されたユーザー情報

    Raises:
        HTTPException: データベースエラー時
    """
    try:
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"ユーザーの作成中にエラーが発生しました: {str(e)}"
        )
