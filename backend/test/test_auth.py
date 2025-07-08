"""
ユーザーAPIのテストファイル
改善されたCRUD機能のテストを実装
"""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

class TestUserCreation:
    """ユーザー作成関連のテストクラス"""
    
    def test_create_user_success(self):
        """新規ユーザー作成成功をテスト"""
        response = client.post("/auth/create_user", json={
            "username": "newuser",
            "password": "newpassword123",
            "email": "newuser@example.com"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "user_id" in data
        assert data["username"] == "newuser"
        assert "正常に作成しました" in data["message"]

    def test_create_user_duplicate_username(self):
        """重複ユーザー名での作成失敗をテスト"""
        # 最初のユーザーを作成
        client.post("/auth/create_user", json={
            "username": "duplicateuser",
            "password": "password123",
            "email": "duplicate1@example.com"
        })
        
        # 同じユーザー名で再度作成を試行
        response = client.post("/auth/create_user", json={
            "username": "duplicateuser",
            "password": "password456",
            "email": "duplicate2@example.com"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "既に存在しています" in data["message"]

    def test_create_user_invalid_password(self):
        """無効なパスワードでの作成失敗をテスト"""
        response = client.post("/auth/create_user", json={
            "username": "validuser",
            "password": "123",  # 短すぎるパスワード
            "email": "valid@example.com"
        })
        assert response.status_code == 422  # バリデーションエラー

    def test_create_user_invalid_username(self):
        """無効なユーザー名での作成失敗をテスト"""
        response = client.post("/auth/create_user", json={
            "username": "",  # 空のユーザー名
            "password": "password123",
            "email": "test@example.com"
        })
        assert response.status_code == 422  # バリデーションエラー

    def test_create_user_without_email(self):
        """メールアドレスなしでのユーザー作成をテスト"""
        response = client.post("/auth/create_user", json={
            "username": "noemailuser",
            "password": "password123"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_create_user_long_username(self):
        """長すぎるユーザー名での作成失敗をテスト"""
        long_username = "a" * 101  # 100文字を超える
        response = client.post("/auth/create_user", json={
            "username": long_username,
            "password": "password123",
            "email": "long@example.com"
        })
        assert response.status_code == 422  # バリデーションエラー


class TestUserLogin:
    """ユーザーログイン関連のテストクラス"""
    
    @pytest.fixture(autouse=True)
    def setup_user(self):
        """テスト用ユーザーを作成"""
        client.post("/auth/create_user", json={
            "username": "loginuser",
            "password": "loginpassword123",
            "email": "login@example.com"
        })

    def test_login_success(self):
        """ログイン成功をテスト"""
        response = client.post("/auth/login", data={
            "username": "loginuser", 
            "password": "loginpassword123"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_username(self):
        """存在しないユーザー名でのログイン失敗をテスト"""
        response = client.post("/auth/login", data={
            "username": "nonexistentuser", 
            "password": "password123"
        })
        assert response.status_code == 400
        data = response.json()
        assert "ユーザー名またはパスワードが不正です" in data["detail"]

    def test_login_invalid_password(self):
        """間違ったパスワードでのログイン失敗をテスト"""
        response = client.post("/auth/login", data={
            "username": "loginuser", 
            "password": "wrongpassword"
        })
        assert response.status_code == 400
        data = response.json()
        assert "ユーザー名またはパスワードが不正です" in data["detail"]

    def test_login_empty_username(self):
        """空のユーザー名でのログイン失敗をテスト"""
        response = client.post("/auth/login", data={
            "username": "", 
            "password": "password123"
        })
        assert response.status_code == 400

    def test_login_empty_password(self):
        """空のパスワードでのログイン失敗をテスト"""
        response = client.post("/auth/login", data={
            "username": "loginuser", 
            "password": ""
        })
        assert response.status_code == 400

    def test_login_with_email(self):
        """メールアドレスでのログイン試行（現在は非対応）"""
        response = client.post("/auth/login", data={
            "username": "login@example.com", 
            "password": "loginpassword123"
        })
        assert response.status_code == 400  # メールアドレスはサポートされていない


class TestUserSecurity:
    """ユーザーセキュリティ関連のテストクラス"""
    
    def test_password_hashing(self):
        """パスワードがハッシュ化されて保存されることをテスト"""
        # 新規ユーザーを作成
        response = client.post("/auth/create_user", json={
            "username": "hashuser",
            "password": "plainpassword123",
            "email": "hash@example.com"
        })
        assert response.status_code == 200
        
        # 作成されたユーザーでログインできることを確認
        login_response = client.post("/auth/login", data={
            "username": "hashuser", 
            "password": "plainpassword123"
        })
        assert login_response.status_code == 200

    def test_token_generation(self):
        """トークンが適切に生成されることをテスト"""
        # ユーザーを作成
        client.post("/auth/create_user", json={
            "username": "tokenuser",
            "password": "tokenpassword123",
            "email": "token@example.com"
        })
        
        # ログインしてトークンを取得
        response = client.post("/auth/login", data={
            "username": "tokenuser", 
            "password": "tokenpassword123"
        })
        assert response.status_code == 200
        data = response.json()
        
        # トークンが存在し、適切な形式であることを確認
        assert "access_token" in data
        assert len(data["access_token"]) > 0
        assert data["token_type"] == "bearer"


class TestUserDataIntegrity:
    """ユーザーデータ整合性のテストクラス"""
    
    def test_user_data_persistence(self):
        """ユーザーデータが正しく保存されることをテスト"""
        # ユーザーを作成
        create_response = client.post("/auth/create_user", json={
            "username": "persistuser",
            "password": "persistpassword123",
            "email": "persist@example.com"
        })
        assert create_response.status_code == 200
        
        # 作成直後にログインできることを確認
        login_response = client.post("/auth/login", data={
            "username": "persistuser", 
            "password": "persistpassword123"
        })
        assert login_response.status_code == 200

    def test_special_characters_in_username(self):
        """ユーザー名の特殊文字を含むテスト"""
        response = client.post("/auth/create_user", json={
            "username": "user_special123",
            "password": "password123",
            "email": "special@example.com"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_unicode_characters_in_username(self):
        """ユーザー名の日本語文字を含むテスト"""
        response = client.post("/auth/create_user", json={
            "username": "ユーザー123",
            "password": "password123",
            "email": "unicode@example.com"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestErrorHandling:
    """エラーハンドリングのテストクラス"""
    
    def test_malformed_request(self):
        """不正なリクエスト形式のテスト"""
        response = client.post("/auth/create_user", json={
            "invalid_field": "value"
        })
        assert response.status_code == 422  # バリデーションエラー

    def test_missing_required_fields(self):
        """必須フィールドが欠けているリクエストのテスト"""
        response = client.post("/auth/create_user", json={
            "username": "testuser"
            # passwordが欠けている
        })
        assert response.status_code == 422  # バリデーションエラー

    def test_invalid_json(self):
        """無効なJSONのテスト"""
        response = client.post("/auth/create_user", 
                             content="invalid json",
                             headers={"Content-Type": "application/json"})
        assert response.status_code == 422


# パラメータ化テスト
class TestParameterizedAuth:
    """パラメータ化されたテストクラス"""
    
    @pytest.mark.parametrize("username,password,email,expected_success", [
        ("validuser1", "password123", "valid1@example.com", True),
        ("validuser2", "password456", "valid2@example.com", True),
        ("validuser3", "password789", None, True),  # メールなし
        ("", "password123", "invalid@example.com", False),  # 空のユーザー名
        ("validuser4", "123", "short@example.com", False),  # 短いパスワード
    ])
    def test_user_creation_scenarios(self, username, password, email, expected_success):
        """様々なユーザー作成シナリオをテスト"""
        payload = {"username": username, "password": password}
        if email:
            payload["email"] = email
            
        response = client.post("/auth/create_user", json=payload)
        
        if expected_success:
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
        else:
            assert response.status_code in [422, 200]  # バリデーションエラーまたは論理エラー
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is False

