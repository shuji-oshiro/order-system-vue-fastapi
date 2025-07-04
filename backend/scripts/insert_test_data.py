"""
テスト用データをデータベースにインサートするスクリプト
test_menudata.csvのデータを参考にして、カテゴリとメニューデータを作成します。
"""

import csv
import os
import sys
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# プロジェクトのルートディレクトリをsys.pathに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# データベース設定（database.pyから直接コピー）
def is_running_under_pytest():
    return "pytest" in sys.modules or os.getenv("PYTEST_CURRENT_TEST") is not None

# backend/dataフォルダが存在しない場合は作成
data_dir = "./data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)

DATABASE_URL = "sqlite:///./data/store_database.db"

# SQLAlchemy が DBと通信するためのエンジンを作成
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# モデル定義（app.models.modelから直接コピー）
from datetime import datetime
from sqlalchemy import String, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

class Category(Base):
    __tablename__ = "categories"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, index=True, unique=True)
    description: Mapped[str] = mapped_column(String, nullable=True)
    
    menu: Mapped[list["Menu"]] = relationship("Menu", back_populates="category")

class Menu(Base):
    __tablename__ = "menus"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"), index=True)
    name: Mapped[str] = mapped_column(String, index=True, unique=True)
    price: Mapped[int] = mapped_column(nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=True)
    search_text: Mapped[str] = mapped_column(String, index=True, nullable=False)
    
    orders: Mapped[list["Order"]] = relationship("Order", back_populates="menu")
    category: Mapped["Category"] = relationship("Category", back_populates="menu")

class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    order_date: Mapped[datetime] = mapped_column(default=func.now())
    seat_id: Mapped[int] = mapped_column(index=True)
    menu_id: Mapped[int] = mapped_column(ForeignKey("menus.id"), index=True)
    order_cnt: Mapped[int] = mapped_column(nullable=False)
    
    menu: Mapped["Menu"] = relationship("Menu", back_populates="orders")


def create_tables():
    """テーブルを作成します"""
    Base.metadata.create_all(bind=engine)
    print("テーブルが作成されました。")


def insert_categories():
    """CSVファイルからカテゴリデータをインサートします"""
    db = SessionLocal()
    try:
        # 既存のカテゴリをチェック
        existing_categories = db.query(Category).all()
        if existing_categories:
            print("カテゴリデータは既に存在します。")
            return

        # CSVファイルのパス
        csv_file_path = project_root / "test" / "testdata" / "testdata_categories.csv"
        
        if not csv_file_path.exists():
            print(f"カテゴリCSVファイルが見つかりません: {csv_file_path}")
            return

        categories_inserted = 0
        
        # CSVファイルを読み込んでカテゴリデータをインサート
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            for row in csv_reader:
                category = Category(
                    id=int(row['id']),
                    name=row['name'],
                    description=row['description']
                )
                db.add(category)
                categories_inserted += 1

        db.commit()
        print(f"{categories_inserted}件のカテゴリをインサートしました。")

    except Exception as e:
        db.rollback()
        print(f"カテゴリインサート中にエラーが発生しました: {e}")
        raise
    finally:
        db.close()


def insert_menus_from_csv():
    """CSVファイルからメニューデータをインサートします"""
    db = SessionLocal()
    try:
        # 既存のメニューをチェック
        existing_menus = db.query(Menu).all()
        if existing_menus:
            print("メニューデータは既に存在します。")
            return

        # CSVファイルのパス
        csv_file_path = project_root / "test" / "testdata" / "testdata_menudata.csv"
        
        if not csv_file_path.exists():
            print(f"メニューCSVファイルが見つかりません: {csv_file_path}")
            return

        menus_inserted = 0
        
        # CSVファイルを読み込んでメニューデータをインサート
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            for row in csv_reader:
                menu = Menu(
                    category_id=int(row['category_id']),
                    name=row['name'],
                    price=int(row['price']),
                    description=row['description'],
                    search_text=row['search_text']
                )
                db.add(menu)
                menus_inserted += 1

        db.commit()
        print(f"{menus_inserted}件のメニューをインサートしました。")

    except Exception as e:
        db.rollback()
        print(f"メニューインサート中にエラーが発生しました: {e}")
        raise
    finally:
        db.close()


def clear_all_data():
    """すべてのデータを削除します（テスト用）"""
    db = SessionLocal()
    try:
        # 外部キー制約があるため、順序に注意して削除
        db.query(Menu).delete()
        db.query(Category).delete()
        db.commit()
        print("すべてのデータを削除しました。")
    except Exception as e:
        db.rollback()
        print(f"データ削除中にエラーが発生しました: {e}")
        raise
    finally:
        db.close()


def main():
    """メイン処理"""
    print("=== テストデータインサートスクリプト ===")
    
    # コマンドライン引数でクリアオプションをチェック
    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        confirm = input("すべてのデータを削除しますか？ (y/N): ")
        if confirm.lower() == 'y':
            clear_all_data()
        return

    try:
        # テーブル作成
        create_tables()
        
        # カテゴリデータをインサート
        insert_categories()
        
        # メニューデータをインサート
        insert_menus_from_csv()
        
        print("=== テストデータのインサートが完了しました ===")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
