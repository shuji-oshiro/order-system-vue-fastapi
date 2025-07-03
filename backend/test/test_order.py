from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

# def test_import_menu():
#     response = client.post("/menu", json={
#         "file_path": "backend/test/test_menudata.csv"
#     })
    
#     assert response.status_code == 200
#     data = response.json()
#     assert len(data) == 14  # CSVからメニューが読み込まれたことを確認

def test_get_orders_error():
    response = client.get("/order/2")
    assert response.status_code == 404  # 存在しないシートIDでのリクエストは404エラーを返すことを確認
    data = response.json()

def test_add_order1():
    response = client.post("/order", json=
        [{
            "seat_id": 1,
            "menu_id": 2,
            "order_cnt": 2
        }]
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1 # 1件の注文が追加されたことを確認

def test_add_order2():
    response = client.post("/order", json=[
        {
            "seat_id": 2,
            "menu_id": 2,
            "order_cnt": 2
        },
        {
            "seat_id": 2,
            "menu_id": 3,
            "order_cnt": 1
        }
    ])
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2 # 2件の注文が追加されたことを確認


def test_get_orders():
    response = client.get("/order/2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2 # 指定したシートの注文が2件のであることを確認

def test_delete_order():
    response = client.delete("/order/1")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 0 # 注文が1件削除されたことを確認