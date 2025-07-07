"""
メニューAPIの改良版テストファイル
強化されたCRUD機能とバリデーション機能のテストを実装
"""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_get_menus_empty():
    """空の状態でメニュー取得をテスト"""
    response = client.get("/menu/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.db_setup(True)
def test_get_menus_with_data():
    """データがある状態でメニュー取得をテスト"""
    response = client.get("/menu/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    
    # 最初のメニューの構造をチェック
    if data:
        menu = data[0]
        assert "id" in menu
        assert "name" in menu
        assert "price" in menu
        assert "category" in menu
        assert "category_id" in menu


@pytest.mark.db_setup(True)
def test_get_menus_with_pagination():
    """ページネーション機能をテスト"""
    # 最初の5件を取得
    response = client.get("/menu/?skip=0&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 5


@pytest.mark.db_setup(True)
def test_get_menu_count():
    """メニュー総数取得をテスト"""
    response = client.get("/menu/count")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert isinstance(data["count"], int)
    assert data["count"] >= 0


@pytest.mark.db_setup(True)
def test_search_menus():
    """メニュー検索機能をテスト"""
    response = client.get("/menu/search?q=唐揚げ")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    
    # 検索結果が正しいかチェック
    for menu in data:
        assert "唐揚げ" in menu["name"] or "唐揚げ" in menu["search_text"]


@pytest.mark.db_setup(True)
def test_get_menu_by_id_success():
    """ID指定でメニュー取得成功のテスト"""
    response = client.get("/menu/1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert "name" in data
    assert "price" in data
    assert "category" in data


def test_get_menu_by_id_not_found():
    """存在しないIDでメニュー取得のテスト"""
    response = client.get("/menu/9999")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


@pytest.mark.db_setup(True)
def test_get_menus_by_category_success():
    """カテゴリ指定でメニュー取得成功のテスト"""
    response = client.get("/menu/category/1")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    
    # 全てのメニューが指定したカテゴリに属するかチェック
    for menu in data:
        assert menu["category_id"] == 1


def test_get_menus_by_category_not_found():
    """存在しないカテゴリIDでメニュー取得のテスト"""
    response = client.get("/menu/category/9999")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


@pytest.mark.db_setup(True)
def test_create_menu_success():
    """単一メニュー作成成功のテスト"""
    new_menu = {
        "category_id": 1,
        "name": "テストメニュー",
        "price": 800,
        "description": "テスト用のメニューです",
        "search_text": "てすとめにゅー"
    }
    
    response = client.post("/menu/", json=new_menu)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == new_menu["name"]
    assert data["price"] == new_menu["price"]
    assert data["category_id"] == new_menu["category_id"]
    assert "id" in data


def test_create_menu_invalid_category():
    """存在しないカテゴリIDでメニュー作成のテスト"""
    new_menu = {
        "category_id": 9999,
        "name": "無効カテゴリメニュー",
        "price": 800,
        "description": "無効なカテゴリ",
        "search_text": "むこうかてごり"
    }
    
    response = client.post("/menu/", json=new_menu)
    assert response.status_code == 400
    data = response.json()
    assert "見つかりません" in data["detail"]


@pytest.mark.db_setup(True)
def test_create_menu_duplicate_name():
    """重複する名前でメニュー作成のテスト"""
    new_menu = {
        "category_id": 1,
        "name": "唐揚げ",  # 既存のメニュー名
        "price": 800,
        "description": "重複テスト",
        "search_text": "じゅうふく"
    }
    
    response = client.post("/menu/", json=new_menu)
    assert response.status_code == 400
    data = response.json()
    assert "既に存在します" in data["detail"]


def test_create_menu_invalid_price():
    """無効な価格でメニュー作成のテスト"""
    new_menu = {
        "category_id": 1,
        "name": "無効価格メニュー",
        "price": -100,  # 負の価格
        "description": "無効な価格",
        "search_text": "むこうかかく"
    }
    
    response = client.post("/menu/", json=new_menu)
    assert response.status_code == 422  # バリデーションエラー


@pytest.mark.db_setup(True)
def test_create_multiple_menus_success():
    """複数メニュー一括作成成功のテスト"""
    menus = [
        {
            "category_id": 1,
            "name": "一括テスト1",
            "price": 600,
            "description": "一括作成テスト1",
            "search_text": "いっかつてすと1"
        },
        {
            "category_id": 2,
            "name": "一括テスト2",
            "price": 900,
            "description": "一括作成テスト2",
            "search_text": "いっかつてすと2"
        }
    ]
    
    request_data = {"menus": menus}
    response = client.post("/menu/bulk", json=request_data)
    assert response.status_code == 201
    data = response.json()
    assert data["created_count"] == 2
    assert len(data["menus"]) == 2


def test_create_multiple_menus_empty_list():
    """空のリストで一括作成のテスト"""
    request_data = {"menus": []}
    response = client.post("/menu/bulk", json=request_data)
    assert response.status_code == 422  # バリデーションエラー


@pytest.mark.db_setup(True)
def test_update_menu_success():
    """メニュー更新成功のテスト"""
    update_data = {
        "name": "更新されたメニュー",
        "price": 1200,
        "description": "更新されたメニューの説明"
    }
    
    response = client.put("/menu/1", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["price"] == update_data["price"]
    assert data["description"] == update_data["description"]
    assert data["id"] == 1


def test_update_menu_not_found():
    """存在しないメニューの更新テスト"""
    update_data = {
        "name": "存在しない",
        "price": 1000
    }
    
    response = client.put("/menu/9999", json=update_data)
    assert response.status_code == 404
    data = response.json()
    assert "見つかりません" in data["detail"]


@pytest.mark.db_setup(True)
def test_update_menu_partial():
    """部分更新のテスト"""
    update_data = {
        "price": 1500  # 価格のみ更新
    }
    
    response = client.put("/menu/1", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["price"] == update_data["price"]
    assert data["id"] == 1


@pytest.mark.db_setup(True)
def test_delete_menu_without_orders():
    """注文がないメニューの削除テスト"""
    # まず新しいメニューを作成
    new_menu = {
        "category_id": 1,
        "name": "削除テストメニュー",
        "price": 500,
        "description": "削除テスト用",
        "search_text": "さくじょてすと"
    }
    create_response = client.post("/menu/", json=new_menu)
    created_menu = create_response.json()
    menu_id = created_menu["id"]
    
    # 削除実行
    response = client.delete(f"/menu/{menu_id}")
    assert response.status_code == 200
    data = response.json()
    assert "削除しました" in data["message"]


def test_delete_menu_not_found():
    """存在しないメニューの削除テスト"""
    response = client.delete("/menu/9999")
    assert response.status_code == 404
    data = response.json()
    assert "見つかりません" in data["detail"]


# CSV インポート機能のテスト（実際のCSVファイルが必要）
def test_csv_import_invalid_file_type():
    """無効なファイル形式でのCSVインポートテスト"""
    # テキストファイルをアップロード
    files = {"file": ("test.txt", "invalid content", "text/plain")}
    response = client.post("/menu/import", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "CSVファイルのみ" in data["detail"]


# 従来互換APIのテスト
@pytest.mark.db_setup(True)
def test_legacy_add_menu():
    """従来互換のメニュー追加APIテスト"""
    new_menu = {
        "category_id": 1,
        "name": "従来互換テスト",
        "price": 700,
        "description": "従来互換テスト",
        "search_text": "じゅうらいごかん"
    }
    
    response = client.put("/menu/", json=new_menu)
    assert response.status_code == 200  # 従来は200を返す
    data = response.json()
    assert isinstance(data, list)  # 従来は全メニューリストを返す


@pytest.mark.db_setup(True)
def test_legacy_update_menu():
    """従来互換のメニュー更新APIテスト"""
    update_data = {
        "menu_id": 1,
        "category_id": 1,
        "name": "従来更新テスト",
        "price": 1100,
        "description": "従来更新テスト",
        "search_text": "じゅうらいこうしん"
    }
    
    response = client.patch("/menu/", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)  # 従来は全メニューリストを返す
