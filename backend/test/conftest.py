# backend/test/conftest.py
import pytest
from pathlib import Path
from sqlalchemy import text
from backend.app.database.database import engine, Base




@pytest.fixture(scope="module", autouse=True)
def setup_test_data(request):
    """各テストファイル用のデータセットアップ"""
    test_file = request.module.__name__
    
    try:
        # SQLファイルでDELETE + INSERTを行うため、事前のリセットは不要
        print(f"{test_file} - テストファイル開始")
        
        yield
        
    finally:
        # 各テストファイル終了後のクリーンアップ（例外発生時も必ず実行）
        print(f"{test_file} - テストデータクリーンアップ")
        try:
            reset_database()
        except Exception as cleanup_error:
            print(f"ファイル終了時のクリーンアップ中にエラーが発生しました: {cleanup_error}")



def cleanup_all_testdata():
    """全テストデータをクリーンアップ（DELETE のみ）"""
    execute_sql_file("cleanup_all.sql")


def setup_categories_testdata():
    """カテゴリテストデータをセットアップ（INSERT）"""
    execute_sql_file("insert_categories.sql")


def setup_menus_testdata():
    """メニューテストデータをセットアップ（INSERT）"""
    execute_sql_file("insert_menus.sql")

def setup_orders_testdata():
    """注文テストデータをセットアップ（INSERT）"""
    execute_sql_file("insert_orders.sql")

def setup_recommend_orders_testdata():
    """レコメンドテスト用注文データをセットアップ（INSERT）"""
    execute_sql_file("insert_recommend_orders.sql")


def execute_sql_file(sql_filename):
    """SQLファイルを読み込んで実行（最もシンプルな方式）"""
    try:
        # testdataフォルダのパスを取得
        test_dir = Path(__file__).parent
        sql_file_path = test_dir / "testdata" / sql_filename
        
        if not sql_file_path.exists():
            print(f"SQLファイルが見つかりません: {sql_file_path}")
            return
        
        # SQLファイルを読み込み
        with open(sql_file_path, 'r', encoding='utf-8') as file:
            sql_content = file.read()
        
        # SQLを実行
        with engine.connect() as connection:
            # セミコロンで分割して複数のSQL文を実行
            sql_statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            for sql_statement in sql_statements:
                if sql_statement:  # 空でない場合のみ実行
                    connection.execute(text(sql_statement))
            
            connection.commit()
            print(f"SQLファイル '{sql_filename}' を実行しました")
        
    except Exception as e:
        print(f"SQLファイル実行中にエラー: {e}")
        raise


# 個別テスト関数用のクリーンアップフィクスチャ
@pytest.fixture(autouse=True)
def test_isolation(request):
    """テスト関数間の分離を提供"""
    test_name = request.function.__name__
    test_file = request.module.__name__

    # カスタムマーカーからデータを取得
    marker = request.node.get_closest_marker("db_setup")
        
    # テスト開始前の準備（SQLファイルがDELETE + INSERTするため、リセット不要）
    if "test_menu" in test_file and test_name != "test_get_orders_error":
        # メニューテストの場合、カテゴリデータが必要
        print(f"{test_name} - カテゴリデータをSQLファイルで準備中")
        cleanup_all_testdata()
        setup_categories_testdata()

        if marker and marker.args:
            # マーカーに引数がある場合はそのデータを使用
            setup_menus_testdata()
    
    elif "test_order" in test_file and test_name != "test_get_orders_error":
        # 注文テストの場合、カテゴリとメニューデータが必要
        print(f"{test_name} - カテゴリ・メニューデータをSQLファイルで準備中")
        cleanup_all_testdata()
        setup_categories_testdata()
        setup_menus_testdata()

        if marker and marker.args:
            # マーカーに引数がある場合はそのデータを使用
            setup_orders_testdata()

    elif "test_recommend" in test_file and test_name != "test_recommend_no_order_data":
        # おすすめメニューテストの場合、カテゴリ、メニュー、レコメンド用注文データが必要
        print(f"{test_name} - おすすめメニューテスト用データをSQLファイルで準備中")
        cleanup_all_testdata()
        setup_categories_testdata()
        setup_menus_testdata()

        if marker and marker.args:
            # マーカーでrecommend_ordersが指定されている場合は専用データを使用
            if "recommend_orders" in marker.args:
                setup_recommend_orders_testdata()
            else:
                setup_orders_testdata()
        else:
            # デフォルトでレコメンド用データを使用
            setup_recommend_orders_testdata()

    elif "test_category" in test_file:
        
        # カテゴリテストの場合、カテゴリデータが必要
        print(f"{test_name} - カテゴリデータをSQLファイルで準備中")
        cleanup_all_testdata()

        if marker and marker.args:
            # マーカーに引数がある場合はそのデータを使用
            setup_categories_testdata()
        if "test_delete_category_with_menus" in test_name:
            # メニューがある場合はメニューデータもセットアップ
            setup_menus_testdata()

    elif "test_auth" in test_file:
        # 認証テストの場合
        pass


    # 特定のテストで個別の前処理が必要な場合
    if "test_get_orders_error" in test_name:
        # 注文がない状態でのテスト用（何もしない）
        pass
    elif "test_recommend_no_order_data" in test_name:
        # 注文データなしでのおすすめメニューテスト用（注文データをセットアップしない）
        cleanup_all_testdata()
        setup_categories_testdata()
        setup_menus_testdata()
        # 注文データはセットアップしない
    
    yield
        
# データベースリセット用フィクスチャ（必要に応じて使用）
@pytest.fixture
def clean_database():
    """明示的にデータベースをクリーンな状態にリセット"""
    # 特定のテストで使用する場合: def test_xxx(clean_database):
    reset_database()
    yield


def reset_database():
    """データベースの全テーブルを削除・再作成してリセット"""
    try:
        # 全テーブルを削除
        Base.metadata.drop_all(bind=engine)
        # 全テーブルを再作成
        Base.metadata.create_all(bind=engine)
        print("データベースをリセットしました")
    except Exception as e:
        print(f"データベースリセット中にエラーが発生しました: {e}")
        # エラーが発生した場合でも、テストを継続させる
        pass


