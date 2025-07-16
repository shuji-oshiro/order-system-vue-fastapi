"""
Neural Collaborative Filtering モデル実装
PyTorchを使用したディープラーニングベースのレコメンドシステム

リファクタリング後: データ処理、学習、推論ロジックを分離
"""
import torch
import logging
import numpy as np
import torch.nn as nn
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from backend.app.ml.data.cache import DataCache
from backend.app.ml.training.trainer import ModelTrainer
from backend.app.ml.data.preprocessing import MenuDataPreprocessor
from backend.app.ml.base.py_torch_base_model import PyTorchBaseModel
from backend.app.ml.inference.predictor import MenuRecommendationPredictor


class NeuralCollaborativeFiltering(PyTorchBaseModel):
    """
    Neural Collaborative Filtering モデル
    メニュー間の関連性を学習する
    特徴量: 共起頻度、時間帯、カテゴリ類似度
    PyTorchを使用して実装
    """
    
    def __init__(
        self, 
        num_menus: Optional[int] = None,
        embedding_dim: int = 50,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        db: Optional[Session] = None
    ):
        super().__init__("MenuRelationshipNetwork")
        
        # 新しいモジュールのインスタンス
        self.data_cache = DataCache()
        self.preprocessor = MenuDataPreprocessor()
        
        # num_menusが指定されていない場合はDBから取得
        if num_menus is None:
            if db is None:
                raise ValueError("num_menusまたはdbセッションのいずれかが必要です")
            from backend.app.crud import menu_crud
            all_menus = menu_crud.get_all_menus(db)
            num_menus = len(all_menus)
            if num_menus == 0:
                raise ValueError("メニューデータが見つかりません")
        
        self.num_menus = num_menus
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 設定を保存（モデル保存時に使用）
        self.config = {
            "num_menus": self.num_menus,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate
        }
        
        # Menu Embeddings
        self.menu1_embedding = nn.Embedding(self.num_menus, self.embedding_dim)
        self.menu2_embedding = nn.Embedding(self.num_menus, self.embedding_dim)
        
        # 特徴量エンコーダー（頻度・時間・カテゴリ類似度用）
        self.feature_encoder = nn.Sequential(
            nn.Linear(3, 16),  # 3つの特徴量（freq, time, category similarity）
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # メインネットワーク（メニューエンベッディング + 特徴量）
        layers = []
        input_dim = self.embedding_dim * 2 + 16  # menu1_emb + menu2_emb + feature_encoded
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            input_dim = hidden_dim
            
        # Output layer（メニュー関連度スコア）
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.main_network = nn.Sequential(*layers)
        
        # 重みの初期化
        self._init_weights()
        
        # Move to device
        self.to(self.device)
        
        logging.info(f"NeuralCollaborativeFilteringモデルを初期化: num_menus={self.num_menus}")
        
    def _init_weights(self):
        """重みの初期化 - Xavier/He初期化を適用"""
        # Menu Embeddings: 正規分布での初期化
        nn.init.normal_(self.menu1_embedding.weight, std=0.01)
        nn.init.normal_(self.menu2_embedding.weight, std=0.01)
        
        # Feature Encoder の重み初期化
        for module in self.feature_encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Main Network の重み初期化
        for module in self.main_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
    def _update_from_config(self, config: Dict[str, Any]):
        """
        保存された設定からモデルを更新（モデル読み込み時に使用）
        
        Args:
            config: 保存されたモデル設定
        """
        logging.info(f"設定からモデルを更新: {config}")
        
        # 動的パラメータを更新
        self.num_menus = config.get('num_menus', self.num_menus)
        self.embedding_dim = config.get('embedding_dim', self.embedding_dim) 
        self.hidden_dims = config.get('hidden_dims', self.hidden_dims)
        self.dropout_rate = config.get('dropout_rate', self.dropout_rate)
        
        # configも更新
        self.config.update(config)
    
    def forward(self, menu1_ids: torch.Tensor, menu2_ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        順伝播: メニュー間の関連度を予測
        
        Args:
            menu1_ids: 基準メニューIDのテンソル
            menu2_ids: 対象メニューIDのテンソル
            features: 特徴量テンソル (freq_similarity, time_similarity, category_similarity)
            
        Returns:
            メニュー間関連度スコア（0-1）
        """
        # メニューエンベッディング
        menu1_emb = self.menu1_embedding(menu1_ids)
        menu2_emb = self.menu2_embedding(menu2_ids)
        
        # 特徴量エンコード
        feature_encoded = self.feature_encoder(features)
        
        # 全てを結合
        combined = torch.cat([menu1_emb, menu2_emb, feature_encoded], dim=1)
        
        # メインネットワーク通過
        output = self.main_network(combined)
        
        return output.squeeze()
    
    def prepare_data(self, db: Optional[Session] = None, force_reload: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        メニュー間の関連性学習用データを準備
        
        Args:
            db: データベースセッション
            force_reload: 強制的にデータを再読み込みするかどうか
            
        Returns:
            menu1_ids, menu2_ids, features, targets, _ のタプル
        """
        logging.info("メニュー関連性データの準備を開始...")
        
        # データキャッシュからオーダーを取得
        orders = self.data_cache.get_cached_orders(db, force_reload)
        
        if not orders:
            raise ValueError("注文データが見つかりません")

        # 前処理モジュールを使用してデータを準備
        menu1_ids, menu2_ids, features, targets, _ = self.preprocessor.prepare_menu_pairs(orders, db)
        
        logging.info("メニュー関連性データの準備完了")
        
        return menu1_ids, menu2_ids, features, targets, _
    
    def fit(self, db: Optional[Session] = None, **kwargs) -> Dict[str, Any]:
        """
        Neural Collaborative Filteringモデルを学習（新しい学習クラスを使用）
        
        Args:
            db: SQLAlchemyデータベースセッション
            **kwargs: 学習パラメータ
                - epochs: 学習エポック数（デフォルト: 100）
                - batch_size: バッチサイズ（デフォルト: 256）
                - learning_rate: 学習率（デフォルト: 0.001）
                - test_size: テストデータの割合（デフォルト: 0.2）
                - patience: 早期停止の許容エポック数（デフォルト: 10）
                - force_reload: 強制的にデータを再読み込み（デフォルト: False）
            
        Returns:
            学習結果の辞書（損失履歴、最終損失、精度等）
        """

        
        # 学習パラメータの設定
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 256)
        learning_rate = kwargs.get('learning_rate', 0.001)
        test_size = kwargs.get('test_size', 0.2)
        patience = kwargs.get('patience', 10)
        force_reload = kwargs.get('force_reload', False)
        
        # データ準備
        menu1_ids, menu2_ids, features, targets, _ = self.prepare_data(db, force_reload=force_reload)
        
        # 訓練・テストデータの分割
        (menu1_train, menu1_test, 
         menu2_train, menu2_test, 
         features_train, features_test,
         targets_train, targets_test) = train_test_split(
            menu1_ids, menu2_ids, features, targets,
            test_size=test_size,
            random_state=42,
            stratify=targets
        )
        
        # PyTorch DataLoaderの作成
        train_dataset = TensorDataset(
            torch.LongTensor(menu1_train),
            torch.LongTensor(menu2_train),
            torch.FloatTensor(features_train),
            torch.FloatTensor(targets_train)
        )
        
        test_dataset = TensorDataset(
            torch.LongTensor(menu1_test),
            torch.LongTensor(menu2_test),
            torch.FloatTensor(features_test),
            torch.FloatTensor(targets_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 損失関数と最適化器の設定
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # 新しいトレーナーを使用して学習
        trainer = ModelTrainer(self, criterion, optimizer, self.device)
        results = trainer.fit(train_loader, test_loader, epochs, patience)
        
        # 学習完了フラグを設定
        self.is_trained = True
        
        logging.info(f"学習完了: 最終損失={results.get('final_val_loss', 0):.4f}")
        return results
    
    def predict_menu_relationships(
        self, 
        base_menu_id: int, 
        candidate_menu_ids: List[int],
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        基準メニューに対する候補メニューとの関連度を予測（新しい推論クラスを使用）
        
        Args:
            base_menu_id: 基準となるメニューID
            candidate_menu_ids: 推薦候補のメニューIDリスト
            db: データベースセッション
            
        Returns:
            関連度スコア付きのメニューリスト
        """
        predictor = MenuRecommendationPredictor(self, self.device)
        return predictor.predict_menu_relationships(base_menu_id, candidate_menu_ids, db)
    
    def recommend_menus(
        self, 
        base_menu_id: int, 
        top_k: int = 5,
        db: Optional[Session] = None,
        exclude_menu_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        基準メニューに対する推薦メニューを取得（新しい推論クラスを使用）
        
        Args:
            base_menu_id: 基準となるメニューID
            top_k: 推薦するメニュー数
            db: データベースセッション
            exclude_menu_ids: 除外するメニューIDのリスト
            
        Returns:
            推薦メニューのリスト
        """
        predictor = MenuRecommendationPredictor(self, self.device)
        return predictor.recommend_menus(base_menu_id, top_k, db, exclude_menu_ids)
    
    def predict(self, user_id: int, menu_id: int, **kwargs) -> float:
        """
        予測を行う（基底クラスの抽象メソッド実装）
        
        Args:
            user_id: ユーザー（座席）ID  
            menu_id: メニューID
            **kwargs: 予測時の追加パラメータ
            
        Returns:
            float: 推薦スコア
        """
        # このモデルではuser_idベースの予測は実装していないため、
        # menu間の関連性予測に置き換える
        db = kwargs.get('db')
        if db is None:
            raise ValueError("予測にはデータベースセッションが必要です")
            
        # 全メニューから候補を取得してmenu_idとの関連性を予測
        relationships = self.predict_menu_relationships(menu_id, [menu_id], db)
        if relationships:
            return relationships[0]['relationship_score']
        return 0.0
    
    def recommend(self, user_id: int, exclude_menu_ids: Optional[List[int]] = None, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        ユーザーに対する推薦を行う（基底クラスの抽象メソッド実装）
        
        Args:
            user_id: ユーザー（座席）ID
            exclude_menu_ids: 除外するメニューIDのリスト  
            top_k: 推薦する件数
            
        Returns:
            List[Tuple[int, float]]: (menu_id, score)のリスト
        """
        # このモデルではuser_idベースの推薦は実装されていないため、
        # 警告を出して空リストを返す
        logging.warning("NeuralCollaborativeFilteringモデルではユーザーベース推薦は実装されていません。recommend_menusメソッドを使用してください。")
        return []
    
    def save_model(self, path: str) -> None:
        """モデルを保存する（基底クラスの抽象メソッド実装）"""
        # ModelLoaderを使用した保存に置き換え
        from backend.app.ml.inference.model_loader import ModelLoader
        loader = ModelLoader()
        import os
        model_name = os.path.basename(os.path.dirname(path))
        loader.save_model(self, model_name)
        self.is_trained = True
        
    def load_model(self, path: str) -> None:
        """モデルを読み込む（基底クラスの抽象メソッド実装）"""
        # ModelLoaderを使用した読み込みに置き換え
        from backend.app.ml.inference.model_loader import ModelLoader
        loader = ModelLoader()
        import os
        model_name = os.path.basename(os.path.dirname(path))
        loader.load_model(self, model_name)
        self.is_trained = True
