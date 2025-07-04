from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_get_menu_allnot_found():
    response = client.get("/menu")
    assert response.status_code == 404

def test_get_menu_by_id_not_found():
    response = client.get("/menu/1")
    assert response.status_code == 404

testDatapath = "backend/test/testdata/testdata_menudata.csv"
def test_import_menu():
    with open(testDatapath, "rb") as f:
        response = client.post("/menu", files={"file": f})

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 50  # CSVからメニューが読み込まれたことを確認

def test_import_menu2():
    with open(testDatapath, "rb") as f:
        response = client.post("/menu", files={"file": f})
    assert response.status_code == 500

def test_add_menu():
    response = client.put("/menu", json={
        "category_id": 1,  # カテゴリIDは適宜設定してください
        "name": "かつ丼",
        "price": 1000,
        "description": "美味しいかつ丼です",
        "search_text": "かつどん"
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 51 # 新しいメニューが追加されたことを確認


def test_get_menus():
    response = client.get("/menu")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list) # レスポンスがリスト型であることを確認


def test_get_menus_byid():
    response = client.get("/menu/1")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list) # レスポンスが辞書型であることを確認


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
    data = response.json()
    assert len(data) == 51 # メニューの件数に変更がないことを確認

def test_delete_menu():
    response = client.delete("/menu/1")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 50  # メニューが削除されたことを確認


def test_get_all_menus_for_category():
    response = client.get("/menulist")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list) # レスポンスがリスト型であることを確認


def test_get_menus_by_category():
    response = client.get("/menu/category/1")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list) # レスポンスがリスト型であることを確認
