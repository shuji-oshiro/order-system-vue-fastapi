import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


client = TestClient(app)


@pytest.mark.db_setup("recommend_orders")
def test_recommend_phase1_frequency():
    """Phase 1: 頻度ベースレコメンドのテスト"""
    # メニューID=11（焼肉定食）に基づくおすすめメニューを取得
    # テストデータでは焼肉定食(11)とビール(21)が最も高頻度で共起
    response = client.get("/recommend/11?phase=1")
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
    
    # 頻度ベースでビール(21)が推薦されることを期待
    assert data["id"] == 21
    assert "ビール" in data["name"]


@pytest.mark.db_setup("recommend_orders")
def test_recommend_phase1_second_frequency():
    """Phase 1: 中頻度パターンのテスト"""
    # メニューID=1（唐揚げ）に基づくおすすめメニューを取得
    # テストデータでは唐揚げ(1)とコーラ(16)が中頻度で共起
    response = client.get("/recommend/1?phase=1")
    assert response.status_code == 200
    data = response.json()
    
    # コーラ(16)が推薦されることを期待
    assert data["id"] == 16
    assert "コーラ" in data["name"]


@pytest.mark.db_setup("recommend_orders")
def test_recommend_phase2_time_category():
    """Phase 2: 時間帯+カテゴリ親和性レコメンドのテスト"""
    response = client.get("/recommend/11?phase=2")
    assert response.status_code == 200
    data = response.json()
    
    # Phase2は基本的にPhase1にフォールバックするので、ビールが推薦される
    assert "id" in data
    assert isinstance(data["id"], int)


@pytest.mark.db_setup("recommend_orders")
def test_recommend_phase3_price_consideration():
    """Phase 3: 価格帯考慮レコメンドのテスト"""
    # メニューID=14（とんかつ定食 950円）の価格帯テスト
    response = client.get("/recommend/14?phase=3")
    assert response.status_code == 200
    data = response.json()
    
    # 価格帯を考慮したレコメンドが返される
    assert "id" in data
    assert isinstance(data["price"], int)
    
    # 価格が基準メニュー（950円）の±30%範囲内であることを確認
    base_price = 950
    price_range = int(base_price * 0.3)  # 285円
    min_price = base_price - price_range  # 665円
    max_price = base_price + price_range  # 1235円
    
    assert min_price <= data["price"] <= max_price


@pytest.mark.db_setup("recommend_orders")
def test_recommend_phase4_complex_scoring():
    """Phase 4: 複合スコアリングシステムのテスト"""
    response = client.get("/recommend/11?phase=4")
    assert response.status_code == 200
    data = response.json()
    
    # 複合スコアリングの結果が返される
    assert "id" in data
    assert "name" in data
    assert "price" in data
    assert isinstance(data["id"], int)


@pytest.mark.db_setup("recommend_orders")
def test_recommend_different_phases():
    """異なるPhaseでの推薦結果の比較テスト"""
    menu_id = 11
    
    # 各Phaseでの推薦結果を取得
    phase1_response = client.get(f"/recommend/{menu_id}?phase=1")
    phase2_response = client.get(f"/recommend/{menu_id}?phase=2")
    phase3_response = client.get(f"/recommend/{menu_id}?phase=3")
    phase4_response = client.get(f"/recommend/{menu_id}?phase=4")
    
    # すべて成功することを確認
    assert phase1_response.status_code == 200
    assert phase2_response.status_code == 200
    assert phase3_response.status_code == 200
    assert phase4_response.status_code == 200
    
    # 各レスポンスが有効な構造を持つことを確認
    for response in [phase1_response, phase2_response, phase3_response, phase4_response]:
        data = response.json()
        assert "id" in data
        assert "name" in data
        assert "price" in data
        assert "category" in data
        # カテゴリの基本構造確認
        category = data["category"]
        assert "name" in category
        assert isinstance(category["name"], str)


@pytest.mark.db_setup("recommend_orders")
def test_recommend_invalid_phase():
    """無効なPhase指定のテスト"""
    # Phase 0（無効）
    response = client.get("/recommend/11?phase=0")
    assert response.status_code == 422  # バリデーションエラー
    
    # Phase 5（無効）
    response = client.get("/recommend/11?phase=5")
    assert response.status_code == 422  # バリデーションエラー


@pytest.mark.db_setup("recommend_orders")
def test_recommend_fallback_mechanism():
    """フォールバック機能のテスト"""
    # 存在しないメニューIDでテスト（高いPhaseから低いPhaseへのフォールバック）
    response = client.get("/recommend/999?phase=4")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


@pytest.mark.db_setup("recommend_orders")
def test_recommend_frequency_validation():
    """頻度ベースレコメンドの検証テスト"""
    # 焼肉定食(11)に対する推薦が期待通りか詳細チェック
    response = client.get("/recommend/11?phase=1")
    assert response.status_code == 200
    data = response.json()
    
    # テストデータに基づいて、ビール(21)が最も頻度が高いはず
    assert data["id"] == 21
    assert data["price"] == 400  # ビールの価格
    assert "ビール" in data["name"]


@pytest.mark.db_setup("recommend_orders")
def test_recommend_category_structure():
    """カテゴリ情報が正しく含まれているかのテスト"""
    response = client.get("/recommend/11?phase=1")
    assert response.status_code == 200
    data = response.json()
    
    # カテゴリ情報の構造確認
    assert "category" in data
    category = data["category"]
    
    # CategoryBaseスキーマに基づく構造確認（idは含まれない）
    assert "name" in category
    assert "description" in category
    assert isinstance(category["name"], str)
    assert isinstance(category["description"], (str, type(None)))  # descriptionはOptional


def test_recommend_no_order_data():
    """注文データが存在しない場合のエラー処理テスト"""
    # 注文データがない状態でテスト
    response = client.get("/recommend/1")
    
    # 注文履歴がない場合は404エラーを返すことを確認
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


@pytest.mark.db_setup("recommend_orders")
def test_recommend_nonexistent_menu():
    """存在しないメニューIDのテスト"""
    response = client.get("/recommend/9999?phase=1")
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    # 実際のエラーメッセージに合わせて修正
    assert "共起するメニューが見つかりません" in data["detail"]


@pytest.mark.db_setup("recommend_orders")
def test_recommend_menu_with_no_cooccurrence():
    """他のメニューと共起履歴がないメニューのテスト"""
    # テストデータに存在するが、他のメニューと一緒に注文されていないメニューがあるかテスト
    # 例: メニューID=3（ポテトフライ）が単独で注文されている場合
    
    # まず存在するメニューで共起データがない場合の動作を確認
    response = client.get("/recommend/3?phase=1")
    # 共起データがない場合は404エラーになるはず
    if response.status_code == 404:
        data = response.json()
        assert "detail" in data
        assert "共起するメニューが見つかりません" in data["detail"]
    else:
        # もし成功した場合は、有効なレコメンドが返されていることを確認
        assert response.status_code == 200
        data = response.json()
        assert "id" in data


@pytest.mark.db_setup("recommend_orders")
def test_recommend_default_phase():
    """デフォルトPhase（Phase 1）のテスト"""
    # Phaseを指定しない場合、デフォルトでPhase 1が使用される
    response = client.get("/recommend/11")
    assert response.status_code == 200
    data = response.json()
    
    # Phase 1の結果と同じになることを確認
    phase1_response = client.get("/recommend/11?phase=1")
    phase1_data = phase1_response.json()
    
    assert data["id"] == phase1_data["id"]
    assert data["name"] == phase1_data["name"]

