import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.app.utils.utils import is_running_under_pytest
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# 通常の実行時は、ファイルベースのSQLiteデータベースを使用
# SQLiteを使用する場合は、ファイルパスを指定します。
database_path = os.getenv("DATABASE_URL", "sqlite:///./backend/data/store_database.db")

# backend/dataフォルダが存在しない場合は作成
data_dir = "./backend/data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)

if is_running_under_pytest():

    # pytest実行中の場合は、メモリ内データベースを使用
    # usersテストで使用するmemory database fileを削除し初期化
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "../../data/.file")
        if os.path.exists(file_path):
            os.remove(file_path)  
        database_path = 'sqlite:///./backend/data/.file:mem1?mode=memory&cache=shared -uri True'
    except Exception as e:
        print(f"Error removing file: {e}")  # ファイル削除エラー
        raise e


# SQLAlchemy が DBと通信するためのエンジンを作成
engine = create_engine(
    # check_same_thread=False：SQLiteでは複数スレッドから接続するために必要な設定（FastAPIは非同期なので必須）
    database_path,
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