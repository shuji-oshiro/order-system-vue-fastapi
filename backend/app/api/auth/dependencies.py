from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from backend.app.api.auth.utils import SECRET_KEY, ALGORITHM
from backend.app.api.auth.auth_schema import UserOut  # ユーザー表示用スキーマ（例）

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme)) -> UserOut:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="認証情報が無効です",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        return UserOut(username=username, email=None, password=None, id=0)  # DBからの取得に置き換えてOK
    except JWTError:
        raise credentials_exception
