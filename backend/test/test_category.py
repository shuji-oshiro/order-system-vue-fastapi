"""
カテゴリAPIのテストファイル
改善されたCRUD機能のテストを実装
"""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_get_categories_empty():
    """空の状態でカテゴリ取得をテスト"""
    response = client.get("/category/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.db_setup(True)
def test_get_categories_with_data():
    """データがある状態でカテゴリ取得をテスト"""
    response = client.get("/category/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    
    # 最初のカテゴリの構造をチェック
    if data:
        category = data[0]
        assert "id" in category
        assert "name" in category
        assert "description" in category


@pytest.mark.db_setup(True)
def test_get_category_by_id_success():
    """ID指定でカテゴリ取得成功のテスト"""
    response = client.get("/category/1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert "name" in data
    assert "description" in data


def test_get_category_by_id_not_found():
    """存在しないIDでカテゴリ取得のテスト"""
    response = client.get("/category/9999")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_create_single_category_success():
    """単一カテゴリ作成成功のテスト"""
    new_category = {
        "name": "テストカテゴリ",
        "description": "テスト用のカテゴリです"
    }
    
    response = client.post("/category/", json=new_category)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == new_category["name"]
    assert data["description"] == new_category["description"]
    assert "id" in data


def test_create_single_category_duplicate_name():
    """重複する名前でカテゴリ作成のテスト"""
    new_category = {
        "name": "テストカテゴリ",
        "description": "重複テスト"
    }
    
    # 1回目は成功
    response1 = client.post("/category/", json=new_category)
    assert response1.status_code == 201
    
    # 2回目は失敗（重複）
    response2 = client.post("/category/", json=new_category)
    assert response2.status_code == 400
    data = response2.json()
    assert "既に存在します" in data["detail"]


def test_create_single_category_invalid_data():
    """無効なデータでカテゴリ作成のテスト"""
    # 名前が空
    invalid_category = {
        "name": "",
        "description": "無効なテスト"
    }
    
    response = client.post("/category/", json=invalid_category)
    assert response.status_code == 422  # バリデーションエラー


def test_create_multiple_categories_success():
    """複数カテゴリ一括作成成功のテスト"""
    categories = [
        {
            "name": "一括テスト1",
            "description": "一括作成テスト1"
        },
        {
            "name": "一括テスト2", 
            "description": "一括作成テスト2"
        }
    ]
    
    response = client.post("/category/bulk", json=categories)
    assert response.status_code == 201
    data = response.json()
    assert len(data) == 2
    assert data[0]["name"] == "一括テスト1"
    assert data[1]["name"] == "一括テスト2"


def test_create_multiple_categories_empty_list():
    """空のリストで一括作成のテスト"""
    response = client.post("/category/bulk", json=[])
    assert response.status_code == 400
    data = response.json()
    assert "空です" in data["detail"]


@pytest.mark.db_setup(True)
def test_update_category_success():
    """カテゴリ更新成功のテスト"""
    update_data = {
        "name": "更新されたカテゴリ",
        "description": "更新されたカテゴリの説明"
    }
    
    response = client.put("/category/1", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["description"] == update_data["description"]
    assert data["id"] == 1


def test_update_category_not_found():
    """存在しないカテゴリの更新テスト"""
    update_data = {
        "name": "存在しない",
        "description": "存在しないカテゴリ"
    }
    
    response = client.put("/category/9999", json=update_data)
    assert response.status_code == 404
    data = response.json()
    assert "見つかりません" in data["detail"]


@pytest.mark.db_setup(True)
def test_delete_category_without_menus():
    """メニューがないカテゴリの削除テスト"""
    # まず新しいカテゴリを作成
    new_category = {
        "name": "削除テスト",
        "description": "削除テスト用"
    }
    create_response = client.post("/category/", json=new_category)
    created_category = create_response.json()
    category_id = created_category["id"]
    
    # 削除実行
    response = client.delete(f"/category/{category_id}")
    assert response.status_code == 200
    data = response.json()
    assert "削除しました" in data["message"]


def test_delete_category_not_found():
    """存在しないカテゴリの削除テスト"""
    response = client.delete("/category/9999")
    assert response.status_code == 404
    data = response.json()
    assert "見つかりません" in data["detail"]


@pytest.mark.db_setup(True) 
def test_delete_category_with_menus():
    """メニューが紐づいているカテゴリの削除テスト"""
    # カテゴリ1にはメニューが紐づいている想定
    response = client.delete("/category/1")
    assert response.status_code == 400
    data = response.json()
    assert "紐づいている" in data["detail"]
