import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.app.utils.utils import is_running_under_pytest


# SQLiteを使用する場合は、ファイルパスを指定します。

DATABASE_URL = "sqlite:///./backend/data/store_database.db"  # ← ファイルで保存
if is_running_under_pytest():
        # usersテストで仕様するmemory database fileを削除する
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "../../data/.file")
        if os.path.exists(file_path):
            os.remove(file_path)  
        DATABASE_URL = 'sqlite:///./backend/data/.file:mem1?mode=memory&cache=shared -uri True'
    except Exception as e:
        print(f"Error removing file: {e}")  # ファイル削除エラー
        raise e
    
# SQLAlchemy が DBと通信するためのエンジンを作成
engine = create_engine(
    # check_same_thread=False：SQLiteでは複数スレッドから接続するために必要な設定（FastAPIは非同期なので必須）
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    
)
# sessionmaker(...)：DBへの操作（SELECT / INSERT など）をするための セッション生成器
# autocommit=False：自動で commit しない（明示的に db.commit() が必要）
# autoflush=False：自動でフラッシュしない（明示的に db.flush() が必要）
# bind=engine：先ほど作成したDBと通信するためのエンジンをバインドする
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# declarative_base()：SQLAlchemyのORMを使うためのベースクラスを生成
# これを継承したクラスがDBモデル（テーブル）となる
Base = declarative_base()

# データベースセッションを取得するための依存関係
# FastAPIの依存性注入を使用して、各エンドポイントでデータベースセッションを取得します。
# これにより、各リクエストごとに新しいセッションが生成され、リクエストが終了したら自動的に閉じられます。
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        print("-------------Closing database session-------------")
        db.close()