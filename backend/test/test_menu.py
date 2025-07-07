import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
client = TestClient(app)

def test_get_allmenus_notfound():
    response = client.get("/menu")
    assert response.status_code == 404

def test_get_menu_byid_notfound():
    response = client.get("/menu/1")
    assert response.status_code == 404

testDatapath = "backend/test/testdata/testdata_menudata.csv"
def test_import_menus_ok():
    with open(testDatapath, "rb") as f:
        response = client.post("/menu", files={"file": f})

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 50  # CSVからメニューが読み込まれたことを確認

@pytest.mark.db_setup(True)
def test_import_menus_error():
    with open(testDatapath, "rb") as f:
        response = client.post("/menu", files={"file": f})
    assert response.status_code == 500

def test_add_menu_ok():
    response = client.put("/menu", json={
        "category_id": 1,  # カテゴリIDは適宜設定してください
        "name": "かつ丼",
        "price": 1000,
        "description": "美味しいかつ丼です",
        "search_text": "かつどん"
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1 # 新しいメニューが追加されたことを確認
    assert isinstance(data, list) # レスポンスがリスト型であることを確認

@pytest.mark.db_setup(True)
def test_add_same_menu_error():
    response = client.put("/menu", json={
        "category_id": 1,  # カテゴリIDは適宜設定してください
        "name": "唐揚げ",
        "price": 1000,
        "description": "美味しいかつ丼です",
        "search_text": "かつどん"
    })
    assert response.status_code == 500  # 同じメニュー名での追加はエラーを返すことを確認
    data = response.json()
    assert "detail" in data  # エラーメッセージが含まれていることを確認

@pytest.mark.db_setup(True)
def test_get_menus_byid_ok():
    response = client.get("/menu/1")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list) # レスポンスが辞書型であることを確認


def test_get_menus_byid_notfound():
    response = client.get("/menu/9999")  # 存在しないメニューIDを指定
    assert response.status_code == 404  # 404エラーを確認

@pytest.mark.db_setup(True)
def test_update_menu():
    response = client.patch("/menu/", json={
        "menu_id": 1,
        "category_id": 1,  # カテゴリIDは適宜設定してください
        "name": "天丼",
        "price": 1200,
        "description": "美味しい天丼です",
        "search_text": "てんどん"
    })
    assert response.status_code == 200

@pytest.mark.db_setup(True)
def test_update_menu_notfound():
    response = client.patch("/menu/", json={
        "menu_id": 9999,  # 存在しないメニューIDを指定
        "category_id": 1,  # カテゴリIDは適宜設定してください
        "name": "天丼",
        "price": 1200,
        "description": "美味しい天丼です",
        "search_text": "てんどん"
    })

    assert response.status_code == 500  # 存在しないメニューIDでの更新はエラーを返すことを確認
    data = response.json()
    assert "detail" in data  # エラーメッセージが含まれていることを確認

@pytest.mark.db_setup(True)
def test_delete_menu():
    response = client.delete("/menu/1")
    assert response.status_code == 200
    data = response.json()
    response.json()
    assert isinstance(data, list) # レスポンスがリスト型であることを確認

@pytest.mark.db_setup(True)
def test_get_all_menus_for_category():
    response = client.get("/menulist")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list) # レスポンスがリスト型であることを確認

@pytest.mark.db_setup(True)
def test_get_menus_by_category():
    response = client.get("/menu/category/1")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list) # レスポンスがリスト型であることを確認
