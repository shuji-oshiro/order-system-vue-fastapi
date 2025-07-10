"""
認証関連のユーティリティ関数

このモジュールは以下の機能を提供します：
1. JWT（JSON Web Token）の生成・検証
2. パスワードのハッシュ化・検証
3. 認証に関する設定値の管理

使用ライブラリ：
- jose: JWT の生成・検証（python-jose）
- passlib: パスワードハッシュ化（bcrypt使用）
"""

from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# JWT設定
# 環境変数から読み取り、デフォルト値を設定
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")  # JWTの署名・検証に使用する秘密鍵
ALGORITHM = os.getenv("ALGORITHM", "HS256")  # JWT署名アルゴリズム（HMAC SHA-256）
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))  # アクセストークンの有効期限（分）

# 開発環境での警告表示
if SECRET_KEY == "your-secret-key":
    print("警告: SECRET_KEYがデフォルト値のままです。本番環境では必ず変更してください。")
    print(".envファイルで SECRET_KEY=強力なランダム文字列 を設定してください。")

# パスワードハッシュ化のコンテキスト
# bcryptアルゴリズムを使用してパスワードを安全にハッシュ化
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict):
    """
    JWTアクセストークンを生成する
    
    Args:
        data (dict): トークンに含めるペイロードデータ
                    通常は {"sub": username} の形式でユーザー名を含む
    
    Returns:
        str: 署名済みJWTトークン文字列
    
    処理の流れ:
    1. 入力データをコピー（元データを変更しないため）
    2. 現在時刻＋有効期限分を計算して有効期限を設定
    3. 有効期限をペイロードに追加（"exp"フィールド）
    4. SECRET_KEYを使用してJWTトークンを署名・生成
    
    例:
        token = create_access_token({"sub": "testuser"})
        # 生成されるトークン: "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
    """
    to_encode = data.copy()  # 元のdataを変更しないようにコピー
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})  # 有効期限をペイロードに追加
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_password(plain_password, hashed_password):
    """
    平文パスワードとハッシュ化パスワードを比較検証する
    
    Args:
        plain_password (str): ユーザーが入力した平文パスワード
        hashed_password (str): データベースに保存されているハッシュ化パスワード
    
    Returns:
        bool: パスワードが一致する場合True、そうでなければFalse
    
    処理の流れ:
    1. passlib（bcrypt）を使用して平文パスワードをハッシュ化
    2. 生成されたハッシュと保存されたハッシュを比較
    3. 一致すればTrue、不一致ならFalseを返す
    
    セキュリティ:
    - bcryptは計算コストが高く、ブルートフォース攻撃に対する耐性が高い
    - saltが自動的に生成され、レインボーテーブル攻撃を防ぐ
    
    例:
        is_valid = verify_password("mypassword", "$2b$12$...")
        # パスワードが正しければTrue
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """
    平文パスワードをハッシュ化する
    
    Args:
        password (str): ハッシュ化したい平文パスワード
    
    Returns:
        str: bcryptアルゴリズムでハッシュ化されたパスワード文字列
    
    処理の流れ:
    1. bcryptアルゴリズムを使用してパスワードをハッシュ化
    2. 自動的にsalt（ランダムな文字列）が生成される
    3. ハッシュ化されたパスワード文字列を返す
    
    セキュリティ特徴:
    - bcryptは適応的ハッシュ関数（計算コストを調整可能）
    - 同じパスワードでも毎回異なるハッシュが生成される（salt使用）
    - ブルートフォース攻撃やレインボーテーブル攻撃に対して高い耐性
    
    例:
        hashed = get_password_hash("mypassword")
        # 結果: "$2b$12$randomsalt.hashedpassword"
        
    使用場面:
    - 新規ユーザー作成時
    - パスワード変更時
    - データベースに保存する前のパスワード処理
    """
    return pwd_context.hash(password)
