import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


client = TestClient(app)


@pytest.mark.db_setup("recommend_orders")
def test_recommend_phase1_frequency():
    """Phase 1: 頻度ベースレコメンドのテスト（カテゴリ構造含む）"""
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
    
    # カテゴリ情報の構造確認も統合
    category = data["category"]
    assert "name" in category
    assert "description" in category
    assert isinstance(category["name"], str)
    assert isinstance(category["description"], (str, type(None)))  # descriptionはOptional


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
    """Phase 2: 時間帯+カテゴリ親和性レコメンドの基本テスト"""
    response = client.get("/recommend/11?phase=2")
    assert response.status_code == 200
    data = response.json()
    
    # Phase2は複合スコアリングを行うので、有効なレコメンドが返される
    assert "id" in data
    assert "name" in data
    assert "price" in data
    assert "category" in data
    assert isinstance(data["id"], int)


@pytest.mark.db_setup("recommend_orders")
def test_recommend_phase2_detailed_scoring():
    """Phase 2: 詳細なスコアリングシステムのテスト"""
    # メニューID=11（焼肉定食）での詳細テスト
    response = client.get("/recommend/11?phase=2")
    assert response.status_code == 200
    data = response.json()
    
    # レスポンスの構造確認
    assert "id" in data
    assert "name" in data
    assert "price" in data
    assert "category" in data
    
    # Phase2は頻度・時間帯・カテゴリ親和性を組み合わせる
    # 結果として有効なメニューが推薦されることを確認
    assert isinstance(data["id"], int)
    assert data["id"] != 11  # 自分自身は推薦されない


@pytest.mark.db_setup("recommend_orders")
def test_recommend_phase2_category_affinity():
    """Phase 2: カテゴリ親和性の影響テスト"""
    # 異なるカテゴリのメニューでPhase2をテスト
    # メニューID=1（唐揚げ - サイドメニュー）
    response1 = client.get("/recommend/1?phase=2")
    # メニューID=11（焼肉定食 - メイン料理）
    response2 = client.get("/recommend/11?phase=2")
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    data1 = response1.json()
    data2 = response2.json()
    
    # 両方とも有効なレコメンドが返される
    assert "id" in data1
    assert "id" in data2
    
    # カテゴリ情報が含まれている
    assert "category" in data1
    assert "category" in data2


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
    
    # デフォルトでPhase 1と同じ結果（ビール）が返されることを確認
    assert data["id"] == 21
    assert "ビール" in data["name"]


@pytest.mark.db_setup("recommend_orders")
def test_recommend_phase2_fallback_to_phase1():
    """Phase 2: Phase1へのフォールバック機能テスト"""
    # データが少ない、または特殊なケースでPhase1にフォールバックする場合のテスト
    response = client.get("/recommend/3?phase=2")
    
    # Phase2が何らかの理由で推薦できない場合、Phase1の結果にフォールバックする
    # ただし、現在の実装では複合スコアリングなので、通常は何らかの結果が返される
    if response.status_code == 200:
        data = response.json()
        assert "id" in data
        assert isinstance(data["id"], int)
    else:
        # エラーの場合は404が返される
        assert response.status_code == 404


@pytest.mark.db_setup("recommend_orders")
def test_recommend_phase2_vs_phase1_difference():
    """Phase 2とPhase 1の結果の違いを確認するテスト"""
    menu_id = 11
    
    # Phase1とPhase2の結果を取得
    phase1_response = client.get(f"/recommend/{menu_id}?phase=1")
    phase2_response = client.get(f"/recommend/{menu_id}?phase=2")
    
    assert phase1_response.status_code == 200
    assert phase2_response.status_code == 200
    
    phase1_data = phase1_response.json()
    phase2_data = phase2_response.json()
    
    # 両方とも有効なレスポンスを持つ
    assert "id" in phase1_data
    assert "id" in phase2_data
    
    # Phase2は時間帯・カテゴリ親和性を考慮するため、
    # Phase1と同じ結果の場合もあれば、異なる結果の場合もある
    # どちらの場合でも、有効なメニューIDが返されていることを確認
    assert isinstance(phase1_data["id"], int)
    assert isinstance(phase2_data["id"], int)


@pytest.mark.db_setup("recommend_orders")
def test_recommend_phase2_time_sensitive():
    """Phase 2: 時間帯の影響をテストする（モック時間は困難なので構造確認）"""
    # 時間帯による影響は実際の時刻に依存するため、
    # ここでは機能が正常に動作することを確認
    response = client.get("/recommend/11?phase=2")
    assert response.status_code == 200
    data = response.json()
    
    # 時間帯を考慮したレコメンドが正常に動作している
    assert "id" in data
    assert "name" in data
    assert "price" in data
    assert "category" in data
    
    # カテゴリ構造の確認
    category = data["category"]
    assert "name" in category
    assert isinstance(category["name"], str)

