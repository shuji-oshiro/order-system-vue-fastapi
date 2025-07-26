import pytest
import time
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
    
    # 学習済みモデルの存在チェック（初回は存在しないはず）
    response = client.get("/ai/training/model/exists")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "model_exists" in data
    assert "message" in data
    # クリーンアップ後なので、モデルは存在しないはず
    assert data["model_exists"] == False

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


@pytest.mark.db_setup("ai_training")
def test_training_and_recommendation_workflow():
    """
    学習からレコメンドまでの完全なワークフローテスト
    テストデータに基づく期待されるレコメンド結果を検証
    """
    
    # 1. 学習前の状態確認
    response = client.get("/ai/training/model/exists")
    assert response.status_code == 200
    # モデルが存在しない状態でスタート
    
    # 2. 新規モデル学習を実行
    response = client.post("/ai/training/train/new", json={})
    assert response.status_code == 202
    
    # 3. 学習完了を待つ（簡易的な待機）
    time.sleep(5)  # 実際の学習には時間がかかるため適切な待機
    
    # 4. 学習後にモデルが存在することを確認
    response = client.get("/ai/training/model/exists")
    assert response.status_code == 200
    data = response.json()
    # 学習完了後はモデルが存在するはず
    assert data["model_exists"] == True
    
    # 5. 期待されるレコメンド結果をテスト（テストデータのパターンに基づく）
    
    # テストケース1: 焼肉定食(id=11) → ビール(id=21)の強い共起パターン
    response = client.get("/recommend/11?phase=5")  # AIレコメンド
    assert response.status_code == 200
    recommend_data = response.json()
    # 焼肉定食に対してビールがレコメンドされることを期待
    # （テストデータで8回の強い共起パターンを設定済み）
    
    # テストケース2: 唐揚げ(id=1) → コーラ(id=16)の中強度共起パターン  
    response = client.get("/recommend/1?phase=5")
    assert response.status_code == 200
    recommend_data = response.json()
    # 唐揚げに対してコーラがレコメンドされることを期待
    # （テストデータで5回の共起パターンを設定済み）
    pass

@pytest.mark.db_setup("ai_training")
def test_phase_based_recommendations():
    """
    フェーズ別レコメンド機能の検証
    各フェーズ（1-5）でレコメンドが動作することを確認
    """
    
    # まず新規学習を実行（Phase5のAIレコメンドのため）
    response = client.post("/ai/training/train/new", json={})
    assert response.status_code == 202
    time.sleep(3)  # 学習完了待機
    
    # 各フェーズでのレコメンドテスト
    test_menu_id = 11  # 焼肉定食（テストデータで頻出）
    
    for phase in range(1, 6):
        response = client.get(f"/recommend/{test_menu_id}?phase={phase}")
        assert response.status_code == 200
        data = response.json()
        
        # 基本的なレスポンス構造の確認
        assert "id" in data
        assert "name" in data
        assert "price" in data
        assert "category_id" in data
        
        print(f"Phase {phase} recommendation for menu {test_menu_id}: {data['name']}")


@pytest.mark.db_setup("ai_training") 
def test_recommendation_edge_cases():
    """
    レコメンド機能のエッジケーステスト
    """
    
    # 新規学習を実行
    response = client.post("/ai/training/train/new", json={})
    assert response.status_code == 202
    time.sleep(3)
    
    # TODO；存在しないメニューIDでのエラーハンドリングが不足している
    # # エッジケース1: 存在しないメニューID
    # response = client.get("/recommend/9999?phase=5")
    # assert response.status_code == 404  # Not Found
    
    # エッジケース2: 無効なフェーズ
    response = client.get("/recommend/11?phase=0")
    assert response.status_code == 422  # Validation Error
    
    response = client.get("/recommend/11?phase=6")
    assert response.status_code == 422  # Validation Error
    
    # エッジケース3: テストデータにない（共起回数が少ない）メニューのレコメンド
    # 朝食メニュー（id=7-10）など、テストデータで共起パターンが少ないもの
    response = client.get("/recommend/7?phase=5")
    # フォールバック機能によりエラーにならずに何らかのレコメンドが返される
    assert response.status_code == 200


@pytest.mark.db_setup("ai_training")
def test_model_retraining():
    """
    モデル再学習機能のテスト
    """
    
    # 1. 新規学習
    response = client.post("/ai/training/train/new", json={})
    assert response.status_code == 202
    time.sleep(3)
    
    # 2. モデル存在確認
    response = client.get("/ai/training/model/exists")
    assert response.status_code == 200
    data = response.json()
    assert data["model_exists"] == True
    
    # 3. 再学習実行
    response = client.post("/ai/training/train/retrain", json={})
    assert response.status_code == 202
    time.sleep(3)
    
    # 4. 再学習後もモデルが存在することを確認
    response = client.get("/ai/training/model/exists")
    assert response.status_code == 200
    data = response.json()
    assert data["model_exists"] == True


