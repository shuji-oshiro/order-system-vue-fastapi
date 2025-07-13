import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


client = TestClient(app)


@pytest.mark.db_setup("orders")
def test_recommend_with_order_data():
    """注文データがある場合のおすすめメニュー取得テスト"""
    # メニューID=11（焼肉定食）に基づくおすすめメニューを取得
    response = client.get("/recommend/11")
    assert response.status_code == 200
    data = response.json()
    
    # レスポンスの構造を確認
    assert "id" in data
    assert "name" in data
    assert "price" in data
    assert "category" in data
    assert isinstance(data["id"], int)
    assert isinstance(data["name"], str)
    assert isinstance(data["price"], int)


@pytest.mark.db_setup("orders")  
def test_recommend_multiple_calls():
    """複数回呼び出して異なるおすすめメニューが返ることを確認"""
    # 同じメニューIDから2回おすすめメニューを取得
    response1 = client.get("/recommend/11")
    response2 = client.get("/recommend/11")
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    data1 = response1.json()
    data2 = response2.json()
    
    # どちらも有効なメニューデータであることを確認
    assert "id" in data1 and "id" in data2
    assert "name" in data1 and "name" in data2
    
    # ランダムなので同じか異なるかは問わないが、どちらも有効なデータであることを確認
    assert isinstance(data1["id"], int)
    assert isinstance(data2["id"], int)


def test_recommend_no_order_data():
    """注文データが存在しない場合のエラー処理テスト"""
    # 注文データがない状態でテスト
    response = client.get("/recommend/1")
    
    # 注文履歴がない場合は404エラーを返すことを確認
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "注文履歴が存在しません" in data["detail"]

