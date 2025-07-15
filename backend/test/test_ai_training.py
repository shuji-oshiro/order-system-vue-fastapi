import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


client = TestClient(app)

@pytest.mark.db_setup("ai_training")
def test_get_model_info():
    
    # モデル情報取得   
    response = client.get("/ai/training/model/info")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "data" in data 

def test_check_model_exists():
    
    # 学習済みモデルの存在チェック
    response = client.get("/ai/training/model/exists")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "model_exists" in data
    assert "message" in data

def test_validate_training_data():
    
    # 学習データの妥当性チェック
    response = client.post("/ai/training/validate")
    assert response.status_code == 200
    data = response.json()    
    assert data["status"] == "success"


def test_train_new_model():
    # 新規モデル学習（最低限の空データを送信）
    response = client.post("/ai/training/train/new", json={})
    
    assert response.status_code == 202  # 非同期処理なので202 Accepted
    data = response.json()
    
    assert data["status"] == "accepted"
    assert "message" in data
    assert data["model_name"] == "neural_cf"


def test_retrain_existing_model():
    # 既存モデルの再学習（最低限の空データを送信）
    response = client.post("/ai/training/train/retrain", json={})
    
    assert response.status_code == 202  # 非同期処理なので202 Accepted
    data = response.json()
    
    assert data["status"] == "accepted"
    assert "message" in data
    assert data["model_name"] == "neural_cf"  # 再学習も同じモデル名を使用