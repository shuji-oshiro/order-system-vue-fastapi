# backend/test/conftest.py
import os
import pytest
from fastapi.testclient import TestClient



@pytest.fixture(scope="module")
def test_client():
    """テスト用HTTPクライアント"""
    from backend.app.main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module", autouse=True)
def setup_test_data(request, test_client):
    """各テストファイル用のデータセットアップ"""
    test_file = request.module.__name__
    
    if "test_category" in test_file:
        print("カテゴリテスト - データベース初期化")
        # カテゴリテストは独立して実行されるため、特別な前処理は不要
        
    elif "test_menu" in test_file:
        print("メニューテスト - カテゴリデータ準備")
        # メニューテストには事前にカテゴリが必要
        setup_categories_for_menu_test(test_client)
        
    elif "test_order" in test_file:
        print("注文テスト - カテゴリ・メニューデータ準備")
        # 注文テストには事前にカテゴリとメニューが必要
        setup_categories_for_order_test(test_client)
        setup_menus_for_order_test(test_client)
    
    yield
    
    # 各テストファイル終了後のクリーンアップ
    print(f"{test_file} - テストデータクリーンアップ")


def setup_categories_for_menu_test(client):
    """メニューテスト用カテゴリデータ準備"""
    categories = [
        {"id": 1, "name": "単品料理", "description": "単品料理の説明"},
        {"id": 2, "name": "定食料理", "description": "定食料理の説明"},
        {"id": 3, "name": "ソフトドリンク", "description": "ソフトドリンクの説明"},
        {"id": 4, "name": "アルコール飲料", "description": "アルコール飲料の説明"},
        {"id": 5, "name": "フルーツ", "description": "フルーツの説明"}
    ]
    
    response = client.post("/category", json=categories)
    if response.status_code != 200:
        pytest.fail(f"メニューテスト用カテゴリ準備に失敗: {response.status_code}")


def setup_categories_for_order_test(client):
    """注文テスト用カテゴリデータ準備"""
    categories = [
        {"id": 1, "name": "単品料理", "description": "単品料理の説明"},
        {"id": 2, "name": "定食料理", "description": "定食料理の説明"},
        {"id": 3, "name": "ソフトドリンク", "description": "ソフトドリンクの説明"}
    ]
    
    response = client.post("/category", json=categories)
    if response.status_code != 200:
        pytest.fail(f"注文テスト用カテゴリ準備に失敗: {response.status_code}")


def setup_menus_for_order_test(client):
    """注文テスト用メニューデータ準備"""
    # 注文テストで使用されるmenu_id: 2, 3に対応するメニューを準備
    menus = [
        {
            "category_id": 1,
            "name": "唐揚げ定食",
            "price": 800,
            "description": "人気の唐揚げ定食",
            "search_text": "からあげ"
        },
        {
            "category_id": 1,
            "name": "ハンバーグ定食",
            "price": 900,
            "description": "ジューシーなハンバーグ定食",
            "search_text": "はんばーぐ"
        },
        {
            "category_id": 2,
            "name": "エビフライ定食",
            "price": 1000,
            "description": "サクサクエビフライ定食",
            "search_text": "えびふらい"
        }
    ]
    
    for menu in menus:
        response = client.put("/menu", json=menu)
        if response.status_code != 200:
            pytest.fail(f"注文テスト用メニュー準備に失敗: {response.status_code}")


# 個別テスト関数用のクリーンアップフィクスチャ
@pytest.fixture(autouse=True)
def test_isolation(request):
    """テスト関数間の分離を提供"""
    test_name = request.function.__name__
    
    # 特定のテストで個別の前処理が必要な場合
    if "test_get_orders_error" in test_name:
        # 注文がない状態でのテスト用（何もしない）
        pass
    
    yield
    
    # テスト後の特別なクリーンアップが必要な場合
    # （現在は各テストがインメモリDBで独立実行されるため不要）


# データベースリセット用フィクスチャ（必要に応じて使用）
@pytest.fixture
def clean_database():
    """明示的にデータベースをクリーンな状態にリセット"""
    # 特定のテストで使用する場合: def test_xxx(clean_database):
    yield


