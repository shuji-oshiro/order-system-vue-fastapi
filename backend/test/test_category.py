from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_add_category():
    response = client.post("/category", json=[
        {
            "id": 1,  
            "name": "単品料理",
            "description": "単品料理の説明"    
        }, 
        {
            "id": 2,  
            "name": "定食料理",
            "description": "定食料理の説明"    
        }, 
        {
            "id": 3,  
            "name": "ソフトドリンク",
            "description": "ソフトドリンクの説明"    
        }, 
        {
            "id": 4,  
            "name": "アルコール飲料",
            "description": "アルコール飲料の説明"    
        }, 
        {
            "id": 5,  
            "name": "フルーツ1",
            "description": "フルーツの説明"    
        }       
    ])
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5 # 1件のカテゴリが追加されたことを確認

def test_get_categories():
    response = client.get("/category")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5 # 5件のカテゴリが取得されたことを確認
