"""
Neural Collaborative Filtering モデル実装
PyTorchを使用したディープラーニングベースのレコメンドシステム
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

from .base_model import PyTorchBaseModel
from backend.app.crud import order_crud
from backend.app.models.model import Menu


class NeuralCollaborativeFiltering(PyTorchBaseModel):
    """
    Neural Collaborative Filtering モデル
    
    ユーザー（座席）とアイテム（メニュー）のEmbeddingを学習し、
    非線形なニューラルネットワークで推薦スコアを予測
    """
    
    def __init__(
        self, 
        num_menus: int,  # num_usersからnum_menusに変更
        embedding_dim: int = 50,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        db: Optional[Session] = None  # DBセッションを初期化時に受け取る
    ):
        super().__init__("MenuRelationshipNetwork")  # 名前変更
        
        self.num_menus = num_menus  # num_usersとnum_itemsを統合
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # データキャッシュ用の変数（インスタンス変数として保持）
        self._cached_orders = None
        self._data_cache_timestamp = None
        self._prepared_data_cache = None
        
        # Menu Embeddings（2つのメニューをエンベッド）
        self.menu1_embedding = nn.Embedding(num_menus, embedding_dim)
        self.menu2_embedding = nn.Embedding(num_menus, embedding_dim)
        
        # 特徴量エンコーダー（頻度・時間・カテゴリ類似度用）
        self.feature_encoder = nn.Sequential(
            nn.Linear(3, 16),  # 3つの特徴量（freq, time, category similarity）
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # メインネットワーク（メニューエンベッディング + 特徴量）
        layers = []
        input_dim = embedding_dim * 2 + 16  # menu1_emb + menu2_emb + feature_encoded
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
            
        # Output layer（メニュー関連度スコア）
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.main_network = nn.Sequential(*layers)
        
        # Initialize embeddings
        self._init_weights()
        
        # Move to device
        self.to(self.device)
        
        # Label encoder for menu IDs
        self.menu_encoder = LabelEncoder()
        
        # 初期化時にデータを読み込む（オプショナル）
        if db is not None:
            self._load_and_cache_data(db)
        
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
        メニュー間の関連性学習用データを準備（キャッシュ機能付き）
        
        Args:
            db: データベースセッション（Noneの場合はキャッシュを使用）
            force_reload: 強制的にデータを再読み込みするかどうか
            
        Returns:
            menu1_ids, menu2_ids, frequencies, time_features, category_features
            (基準メニュー, 関連メニュー, 共起頻度, 時間帯特徴量, カテゴリ特徴量)
        """
        logging.info("メニュー関連性データの準備を開始...")
        
        # ====================
        # キャッシュからのデータ取得戦略
        # ====================
        orders = None
        
        # 1. キャッシュが有効で強制再読み込みでない場合
        if not force_reload and self._is_cache_valid() and self._cached_orders is not None:
            orders = self._cached_orders
            logging.info(f"キャッシュからデータを取得: {len(orders)}件")
        
        # 2. キャッシュが無効またはDBセッションが提供された場合
        elif db is not None:
            orders = order_crud.get_all_orders(db)
            # 新しいデータをキャッシュに保存
            self._cached_orders = orders
            import datetime
            self._data_cache_timestamp = datetime.datetime.now()
            logging.info(f"データベースから新規取得してキャッシュ: {len(orders)}件")
        
        # 3. キャッシュも無効でDBセッションも無い場合
        else:
            if self._cached_orders is not None:
                orders = self._cached_orders
                logging.warning("期限切れキャッシュを使用（DBセッションが提供されていません）")
            else:
                raise ValueError("データが利用できません。DBセッションを提供するか、事前にデータをキャッシュしてください。")
        
        if not orders:
            raise ValueError("注文データが見つかりません")

        # メニュー情報を取得（カテゴリ情報含む）
        if db is not None:
            from backend.app.models.model import Menu
            all_menus = db.query(Menu).all()
            menu_categories = {menu.menu_id: menu.category_id for menu in all_menus}
        else:
            # キャッシュからメニュー情報を復元（簡易版）
            menu_categories = {}
            for order in orders:
                if hasattr(order, 'menu') and order.menu:
                    menu_categories[order.menu_id] = order.menu.category_id

        # DataFrameに変換
        order_data = []
        for order in orders:
            order_data.append({
                'seat_id': order.seat_id,
                'menu_id': order.menu_id,
                'quantity': order.order_cnt,
                'order_datetime': order.order_date,
                'hour': order.order_date.hour if order.order_date else 12,  # デフォルト12時
                'category_id': menu_categories.get(order.menu_id, 0)  # デフォルトカテゴリ0
            })
        
        df = pd.DataFrame(order_data)
        
        # 座席IDごとに注文されたメニューをグループ化
        seat_menus = df.groupby('seat_id')['menu_id'].apply(list).to_dict()
        
        # メニュー間の共起関係を構築
        menu_pairs = []
        
        for seat_id, menu_list in seat_menus.items():
            unique_menus = list(set(menu_list))  # 重複除去
            
            # 同一座席で注文されたメニューのペアを作成
            for i, menu1 in enumerate(unique_menus):
                for j, menu2 in enumerate(unique_menus):
                    if i != j:  # 異なるメニュー同士
                        # メニュー1の特徴量
                        menu1_orders = df[(df['seat_id'] == seat_id) & (df['menu_id'] == menu1)]
                        menu1_freq = len(menu1_orders)
                        menu1_avg_hour = menu1_orders['hour'].mean()
                        menu1_category = menu1_orders['category_id'].iloc[0]
                        
                        # メニュー2の特徴量
                        menu2_orders = df[(df['seat_id'] == seat_id) & (df['menu_id'] == menu2)]
                        menu2_freq = len(menu2_orders)
                        menu2_avg_hour = menu2_orders['hour'].mean()
                        menu2_category = menu2_orders['category_id'].iloc[0]
                        
                        menu_pairs.append({
                            'menu1_id': menu1,
                            'menu2_id': menu2,
                            'co_occurrence': 1,  # 共起フラグ
                            'menu1_freq': menu1_freq,
                            'menu2_freq': menu2_freq,
                            'freq_similarity': min(menu1_freq, menu2_freq) / max(menu1_freq, menu2_freq),
                            'time_similarity': 1.0 - abs(menu1_avg_hour - menu2_avg_hour) / 24.0,
                            'category_similarity': 1.0 if menu1_category == menu2_category else 0.0,
                        })
        
        # ネガティブサンプル（共起しないメニューペア）を生成
        all_menus = df['menu_id'].unique()
        negative_pairs = []
        
        # ポジティブサンプルの2倍のネガティブサンプルを生成
        positive_pairs_set = set((pair['menu1_id'], pair['menu2_id']) for pair in menu_pairs)
        
        import random
        neg_samples_needed = len(menu_pairs) * 2
        attempts = 0
        max_attempts = neg_samples_needed * 5
        
        while len(negative_pairs) < neg_samples_needed and attempts < max_attempts:
            menu1 = random.choice(all_menus)
            menu2 = random.choice(all_menus)
            
            if menu1 != menu2 and (menu1, menu2) not in positive_pairs_set:
                # ランダムペアの特徴量（平均値を使用）
                menu1_data = df[df['menu_id'] == menu1]
                menu2_data = df[df['menu_id'] == menu2]
                
                if len(menu1_data) > 0 and len(menu2_data) > 0:
                    menu1_freq = len(menu1_data)
                    menu2_freq = len(menu2_data)
                    menu1_avg_hour = menu1_data['hour'].mean()
                    menu2_avg_hour = menu2_data['hour'].mean()
                    menu1_category = menu1_data['category_id'].iloc[0]
                    menu2_category = menu2_data['category_id'].iloc[0]
                    
                    negative_pairs.append({
                        'menu1_id': menu1,
                        'menu2_id': menu2,
                        'co_occurrence': 0,  # 非共起フラグ
                        'menu1_freq': menu1_freq,
                        'menu2_freq': menu2_freq,
                        'freq_similarity': min(menu1_freq, menu2_freq) / max(menu1_freq, menu2_freq),
                        'time_similarity': 1.0 - abs(menu1_avg_hour - menu2_avg_hour) / 24.0,
                        'category_similarity': 1.0 if menu1_category == menu2_category else 0.0,
                    })
            
            attempts += 1
        
        # 全データを結合
        all_pairs = menu_pairs + negative_pairs
        pairs_df = pd.DataFrame(all_pairs)
        
        # メニューIDをエンコード
        unique_menus = sorted(df['menu_id'].unique())
        self.menu_encoder = LabelEncoder()
        self.menu_encoder.fit(unique_menus)
        
        pairs_df['menu1_encoded'] = self.menu_encoder.transform(pairs_df['menu1_id'])
        pairs_df['menu2_encoded'] = self.menu_encoder.transform(pairs_df['menu2_id'])
        
        # データをシャッフル
        pairs_df = pairs_df.sample(frac=1).reset_index(drop=True)
        
        logging.info(f"メニューペア準備完了: ポジティブサンプル {len(menu_pairs)}件, ネガティブサンプル {len(negative_pairs)}件")
        logging.info(f"総メニュー数: {len(unique_menus)}")
        
        return (
            pairs_df['menu1_encoded'].values.astype(np.int64),
            pairs_df['menu2_encoded'].values.astype(np.int64),
            pairs_df[['freq_similarity', 'time_similarity', 'category_similarity']].values.astype(np.float32),
            pairs_df['co_occurrence'].values.astype(np.float32),
            np.zeros_like(pairs_df['co_occurrence'].values.astype(np.float32))  # 予備用配列
        )
    
    def fit(self, db: Optional[Session] = None, **kwargs) -> Dict[str, Any]:
        """
        Neural Collaborative Filteringモデルを学習
        
        Args:
            db: SQLAlchemyデータベースセッション（Noneの場合はキャッシュを使用）
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
        # ====================
        # 1. 学習パラメータの設定
        # ====================
        epochs = kwargs.get('epochs', 100)           # 最大学習エポック数
        batch_size = kwargs.get('batch_size', 256)   # 1バッチあたりのサンプル数
        learning_rate = kwargs.get('learning_rate', 0.001)  # Adam最適化器の学習率
        test_size = kwargs.get('test_size', 0.2)     # 全データの20%をテストデータに
        force_reload = kwargs.get('force_reload', False)  # データ強制再読み込み
        
        # ====================
        # 2. データキャッシュからの学習データ準備（効率化）
        # ====================
        # 注文履歴からメニュー間の関連性データを生成
        # キャッシュ機能により初回以降は高速化
        menu1_ids, menu2_ids, features, targets, _ = self.prepare_data(db, force_reload=force_reload)
        
        # ====================
        # 3. 訓練・テストデータの分割
        # ====================
        # 層化分割で0/1ラベルのバランスを保持しながらデータを分割
        (menu1_train, menu1_test, 
         menu2_train, menu2_test, 
         features_train, features_test,
         targets_train, targets_test) = train_test_split(
            menu1_ids, menu2_ids, features, targets,
            test_size=test_size,        # 20%をテストデータに
            random_state=42,            # 再現性のためのシード固定
            stratify=targets            # 0/1ラベルの比率を保持
        )
        
        # ====================
        # 4. PyTorch DataLoaderの作成
        # ====================
        # NumPy配列からPyTorchテンソルに変換してDatasetを作成
        train_dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(menu1_train),     # 基準メニューID（整数型）
            torch.LongTensor(menu2_train),     # 対象メニューID（整数型）
            torch.FloatTensor(features_train), # 特徴量（freq, time, category similarity）
            torch.FloatTensor(targets_train)   # ターゲット（0 or 1の浮動小数点型）
        )
        
        test_dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(menu1_test),
            torch.LongTensor(menu2_test),
            torch.FloatTensor(features_test),
            torch.FloatTensor(targets_test)
        )
        
        # データローダー作成（効率的なバッチ処理のため）
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True    # 訓練データはエポック毎にシャッフル
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False   # テストデータは順序固定
        )
        
        # データローダーの検証
        if len(train_loader) == 0:
            raise ValueError("訓練データローダーが空です。データを確認してください。")
        if len(test_loader) == 0:
            raise ValueError("テストデータローダーが空です。データを確認してください。")
        
        logging.info(f"データローダー作成完了 - 訓練: {len(train_loader)}バッチ, テスト: {len(test_loader)}バッチ")
        
        # ====================
        # 5. 損失関数と最適化器の設定
        # ====================
        # 二値分類のためのBinary Cross Entropy Loss
        criterion = nn.BCELoss()
        # Adam最適化器（適応的学習率、モーメンタム付き）
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # ====================
        # 6. 学習ループの初期化
        # ====================
        train_losses = []  # 各エポックの訓練損失を記録
        test_losses = []   # 各エポックのテスト損失を記録
        
        # 早期停止機構のパラメータ
        best_val_loss = float('inf')                # 最良のバリデーション損失
        patience = kwargs.get('patience', 10)       # 改善しないエポック数の上限
        patience_counter = 0                        # 連続で改善しないエポック数
        
        logging.info(f"学習開始: エポック数={epochs}, バッチサイズ={batch_size}")
        logging.info(f"訓練データサイズ: {len(train_loader)} バッチ, テストデータサイズ: {len(test_loader)} バッチ")
        
        # ====================
        # 7. メインの学習ループ
        # ====================
        for epoch in range(epochs):
            logging.info(f"=== エポック {epoch}/{epochs-1} 開始 ===")
            
            # ====================
            # 7-1. 訓練フェーズ
            # ====================
            nn.Module.train(self, True)  # モデルを学習モードに（Dropout有効化）
            train_loss = 0.0       # このエポックの訓練損失累積値
            batch_count = 0        # バッチカウンター（デバッグ用）
            
            # 訓練データの全バッチを処理
            for menu1_batch, menu2_batch, features_batch, targets_batch in train_loader:
                batch_count += 1
                
                # 最初のエポックは詳細ログ、その後は間隔を空ける
                if epoch == 0 or batch_count % 20 == 0:
                    logging.info(f"  エポック {epoch}, バッチ {batch_count}/{len(train_loader)} 処理中...")
                
                # データの形状チェック（最初のバッチのみ）
                if epoch == 0 and batch_count == 1:
                    logging.info(f"  バッチサイズ: menu1={menu1_batch.shape}, menu2={menu2_batch.shape}")
                    logging.info(f"  特徴量={features_batch.shape}, ターゲット={targets_batch.shape}")
                
                # GPU/CPUにデータを移動
                menu1_batch = self.to_device(menu1_batch)
                menu2_batch = self.to_device(menu2_batch)
                features_batch = self.to_device(features_batch)
                targets_batch = self.to_device(targets_batch)
                
                # 勾配をゼロクリア（前のバッチの勾配をリセット）
                optimizer.zero_grad()
                
                # フォワードプロパゲーション（予測）
                predictions = self.forward(menu1_batch, menu2_batch, features_batch)
                # 損失計算（予測値と実際値の差）
                loss = criterion(predictions, targets_batch)
                
                # デバッグ情報（最初のバッチのみ）
                if batch_count == 1:
                    logging.info(f"最初のバッチ - 予測値範囲: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
                    logging.info(f"最初のバッチ - 実際値範囲: [{targets_batch.min().item():.4f}, {targets_batch.max().item():.4f}]")
                    logging.info(f"最初のバッチ - 損失: {loss.item():.4f}")
                
                # バックプロパゲーション（勾配計算）
                loss.backward()
                # パラメータ更新
                optimizer.step()
                
                # 損失を累積（平均計算のため）
                train_loss += loss.item()
                
                # NaNやInfのチェック
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"損失が異常値です: {loss.item()}")
                    raise ValueError("学習中に異常な損失値が発生しました")
            
            # ====================
            # 7-2. バリデーションフェーズ
            # ====================
            self.eval()        # モデルを評価モードに（Dropout無効化）
            test_loss = 0.0    # このエポックのテスト損失累積値
            
            # 勾配計算を無効化してメモリ節約
            with torch.no_grad():
                # テストデータの全バッチを処理
                for menu1_batch, menu2_batch, features_batch, targets_batch in test_loader:
                    # GPU/CPUにデータを移動
                    menu1_batch = self.to_device(menu1_batch)
                    menu2_batch = self.to_device(menu2_batch)
                    features_batch = self.to_device(features_batch)
                    targets_batch = self.to_device(targets_batch)
                    
                    # 予測と損失計算（勾配計算なし）
                    predictions = self.forward(menu1_batch, menu2_batch, features_batch)
                    loss = criterion(predictions, targets_batch)
                    
                    # テスト損失を累積
                    test_loss += loss.item()
            
            # ====================
            # 7-3. エポック終了処理
            # ====================
            # 平均損失を計算（総損失 ÷ バッチ数）
            avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
            
            # 損失履歴に記録
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            
            # デバッグ情報：毎エポック詳細ログ
            logging.info(f"=== エポック {epoch}/{epochs-1} 完了 ===")
            logging.info(f"  訓練損失: {avg_train_loss:.6f} (総バッチ数: {batch_count})")
            logging.info(f"  テスト損失: {avg_test_loss:.6f}")
            logging.info(f"  処理時間: バッチあたり平均処理")
            
            # 損失が0の場合の警告
            if avg_train_loss == 0.0:
                logging.warning(f"⚠️  エポック {epoch}: 訓練損失が0です。データまたはモデルに問題がある可能性があります。")
            
            if avg_test_loss == 0.0:
                logging.warning(f"⚠️  エポック {epoch}: テスト損失が0です。データまたはモデルに問題がある可能性があります。")
            
            # ====================
            # 7-4. 早期停止の判定
            # ====================
            if avg_test_loss < best_val_loss:
                # テスト損失が改善した場合
                best_val_loss = avg_test_loss
                patience_counter = 0  # カウンターをリセット
                logging.info(f"Epoch {epoch}: バリデーション損失が改善しました ({best_val_loss:.6f})")
            else:
                # テスト損失が改善しなかった場合
                patience_counter += 1
                logging.info(f"Epoch {epoch}: バリデーション損失が改善せず ({patience_counter}/{patience})")
            
            # 早期停止の実行
            if patience_counter >= patience:
                logging.info(f"早期停止: エポック {epoch} で学習終了")
                break
        
        # ====================
        # 8. 学習完了処理
        # ====================
        self.is_trained = True  # 学習済みフラグを設定
        
        # エンコーダーの状態検証
        if not self._validate_encoders():
            logging.warning("エンコーダーの検証に失敗しましたが、学習は完了しました")
        
        logging.info("学習完了")
        
        # 最終的な評価指標を計算（精度、最終損失など）
        final_metrics = self._calculate_metrics(test_loader, criterion)
        
        # ====================
        # 9. 学習結果の返却
        # ====================
        return {
            'train_losses': train_losses,              # 各エポックの訓練損失リスト
            'test_losses': test_losses,                # 各エポックのテスト損失リスト
            'final_train_loss': train_losses[-1],      # 最終訓練損失
            'final_test_loss': test_losses[-1],        # 最終テスト損失
            'best_val_loss': best_val_loss,            # 最良のバリデーション損失
            'epochs_trained': epoch + 1,               # 実際に学習したエポック数
            'early_stopped': patience_counter >= patience,  # 早期停止したかどうか
            **final_metrics                            # 精度などの追加指標
        }
    
    def _calculate_metrics(self, test_loader, criterion):
        """
        追加の評価指標を計算
        
        Args:
            test_loader: テストデータのDataLoader
            criterion: 損失関数（BCELoss）
            
        Returns:
            評価指標の辞書（精度、最終テスト損失）
        """
        self.eval()  # 評価モードに設定（Dropout無効化）
        total_loss = 0.0        # 総損失
        correct_predictions = 0  # 正解数
        total_predictions = 0   # 総予測数
        
        # 勾配計算を無効化してメモリ節約
        with torch.no_grad():
            for menu1_batch, menu2_batch, features_batch, targets_batch in test_loader:
                # データをデバイスに移動
                menu1_batch = self.to_device(menu1_batch)
                menu2_batch = self.to_device(menu2_batch)
                features_batch = self.to_device(features_batch)
                targets_batch = self.to_device(targets_batch)
                
                # 予測実行
                predictions = self.forward(menu1_batch, menu2_batch, features_batch)
                # 損失計算
                loss = criterion(predictions, targets_batch)
                total_loss += loss.item()
                
                # Binary accuracy計算（閾値0.5で二値分類）
                binary_predictions = (predictions > 0.5).float()  # 0.5以上なら1、未満なら0
                correct_predictions += (binary_predictions == targets_batch).sum().item()
                total_predictions += targets_batch.size(0)
        
        # 精度計算（正解数 ÷ 総数）
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'test_accuracy': accuracy,                    # テスト精度（0-1）
            'test_loss_final': total_loss / len(test_loader)  # 平均テスト損失
        }
    
    def predict_menu_relationship(self, menu1_id: int, menu2_id: int, **kwargs) -> float:
        """
        2つのメニュー間の関連度を予測
        
        Args:
            menu1_id: 基準メニューID
            menu2_id: 対象メニューID
            **kwargs: 予測パラメータ（将来の拡張用）
            
        Returns:
            メニュー間関連度スコア (0-1)
        """
        # 学習済み状態の確認
        if not self.is_trained:
            raise ValueError("モデルが学習されていません。fit()メソッドを先に実行してください。")
        
        # エンコーダーの学習状態確認
        if not hasattr(self.menu_encoder, 'classes_'):
            raise ValueError("メニューエンコーダーが学習されていません。prepare_data()が正しく実行されていない可能性があります。")
        
        try:
            # メニューIDをエンコード
            menu1_encoded_array = self.menu_encoder.transform([menu1_id])
            menu2_encoded_array = self.menu_encoder.transform([menu2_id])
            
            # numpy配列から値を取得
            menu1_encoded = int(np.array(menu1_encoded_array)[0])
            menu2_encoded = int(np.array(menu2_encoded_array)[0])
            
            # デフォルト特徴量（推論時は平均的な値を使用）
            default_features = torch.FloatTensor([[0.5, 0.5, 0.5]])  # freq_sim, time_sim, category_sim
            
            # テンソルに変換
            menu1_tensor = torch.LongTensor([menu1_encoded]).to(self.device)
            menu2_tensor = torch.LongTensor([menu2_encoded]).to(self.device)
            features_tensor = default_features.to(self.device)
            
            # 予測実行（評価モードで確実に実行）
            self.eval()  # 評価モードに設定（Dropout無効化）
            with torch.no_grad():
                score = self.forward(menu1_tensor, menu2_tensor, features_tensor)
                return float(score.cpu().numpy())
        
        except (ValueError, KeyError) as e:
            # 未知のIDの場合は詳細なログと平均スコアを返す
            logging.warning(f"未知のメニューID detected - menu1_id: {menu1_id}, menu2_id: {menu2_id}, error: {str(e)}")
            return 0.5
    
    def recommend_similar_menus(self, base_menu_id: int, exclude_menu_ids: Optional[List[int]] = None, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        基準メニューに類似したメニューを推薦
        
        Args:
            base_menu_id: 基準となるメニューID
            exclude_menu_ids: 除外するメニューIDのリスト（デフォルト: None）
            top_k: 推薦する件数（デフォルト: 10）
            
        Returns:
            (menu_id, score)のリスト（スコア降順）
        """
        # 学習済み状態の確認
        if not self.is_trained:
            raise ValueError("モデルが学習されていません。fit()メソッドを先に実行してください。")
        
        # エンコーダーの学習状態確認
        if not hasattr(self.menu_encoder, 'classes_'):
            raise ValueError("メニューエンコーダーが学習されていません。prepare_data()が正しく実行されていない可能性があります。")
        
        if exclude_menu_ids is None:
            exclude_menu_ids = []
        
        # 基準メニューも除外リストに追加（自分自身は推薦しない）
        exclude_menu_ids = list(exclude_menu_ids) + [base_menu_id]
        
        # 全メニューに対するスコアを計算
        all_menus = self.menu_encoder.classes_
        recommendations = []
        
        logging.info(f"メニュー推薦計算開始 - 基準メニュー: {base_menu_id}, 対象メニュー数: {len(all_menus)}")
        
        for menu_id in all_menus:
            if menu_id not in exclude_menu_ids:
                try:
                    score = self.predict_menu_relationship(base_menu_id, menu_id)
                    recommendations.append((int(menu_id), float(score)))
                except Exception as e:
                    # 個別のメニューで予測エラーが発生した場合はスキップ
                    logging.warning(f"メニューID {menu_id} の予測でエラー: {str(e)}")
                    continue
        
        # スコア降順でソート
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        logging.info(f"メニュー推薦計算完了 - 推薦候補数: {len(recommendations)}, 返却数: {min(top_k, len(recommendations))}")
        
        return recommendations[:top_k]
    
    def predict(self, user_id: int, menu_id: int, **kwargs) -> float:
        """
        後方互換性のための予測メソッド（非推奨）
        新しいアーキテクチャでは predict_menu_relationship を使用
        
        Args:
            user_id: 座席ID（非推奨パラメータ、メニューIDとして解釈）
            menu_id: メニューID
            **kwargs: 予測パラメータ
            
        Returns:
            メニュー間関連度スコア (0-1)
        """
        # 警告メッセージ
        logging.warning("predict()は非推奨です。predict_menu_relationship()を使用してください。")
        
        # user_idを基準メニューIDとして扱う
        return self.predict_menu_relationship(user_id, menu_id, **kwargs)
    
    def recommend(self, user_id: int, exclude_menu_ids: Optional[List[int]] = None, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        後方互換性のための推薦メソッド（非推奨）
        新しいアーキテクチャでは recommend_similar_menus を使用
        
        Args:
            user_id: 座席ID（非推奨パラメータ、基準メニューIDとして解釈）
            exclude_menu_ids: 除外するメニューIDのリスト
            top_k: 推薦する件数
            
        Returns:
            (menu_id, score)のリスト（スコア降順）
        """
        # 警告メッセージ
        logging.warning("recommend()は非推奨です。recommend_similar_menus()を使用してください。")
        
        # user_idを基準メニューIDとして扱う
        return self.recommend_similar_menus(user_id, exclude_menu_ids, top_k)
    
    def _validate_encoders(self) -> bool:
        """
        メニューエンコーダーの学習状態を検証
        
        Returns:
            bool: エンコーダーが正しく学習されている場合True
        """
        try:
            # menu_encoderの状態確認
            if not hasattr(self.menu_encoder, 'classes_'):
                logging.error("menu_encoderが学習されていません")
                return False
            
            # クラス数とモデルの次元数の整合性確認
            if len(self.menu_encoder.classes_) > self.num_menus:
                logging.warning(f"menu_encoder classes ({len(self.menu_encoder.classes_)}) > num_menus ({self.num_menus})")
            
            logging.info(f"メニューエンコーダー検証完了 - メニュー数: {len(self.menu_encoder.classes_)}")
            return True
            
        except Exception as e:
            logging.error(f"エンコーダー検証中にエラー: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        モデルの状態情報を取得
        
        Returns:
            モデル情報の辞書
        """
        info = {
            'is_trained': self.is_trained,
            'model_name': self.model_name,
            'device': str(self.device),
            'embedding_dim': self.embedding_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'num_menus': self.num_menus,
            'architecture': 'MenuRelationshipNetwork'
        }
        
        # メニューエンコーダー情報
        if hasattr(self.menu_encoder, 'classes_'):
            info['encoded_menus'] = len(self.menu_encoder.classes_)
            info['menu_ids_range'] = f"{self.menu_encoder.classes_.min()}-{self.menu_encoder.classes_.max()}"
        else:
            info['encoded_menus'] = 0
            info['menu_ids_range'] = "未設定"
            
        return info
    
    def _load_and_cache_data(self, db: Session) -> None:
        """
        データベースから注文データを読み込んでキャッシュ
        
        Args:
            db: データベースセッション
        """
        import datetime
        
        logging.info("注文データのキャッシュを開始...")
        
        # 注文履歴を取得してキャッシュ
        self._cached_orders = order_crud.get_all_orders(db)
        self._data_cache_timestamp = datetime.datetime.now()
        
        if not self._cached_orders:
            logging.warning("注文データが見つかりません")
            return
        
        logging.info(f"注文データキャッシュ完了: {len(self._cached_orders)}件")
    
    def _is_cache_valid(self, max_age_minutes: int = 60) -> bool:
        """
        キャッシュが有効かどうかを確認
        
        Args:
            max_age_minutes: キャッシュの有効期限（分）
            
        Returns:
            bool: キャッシュが有効な場合True
        """
        if self._cached_orders is None or self._data_cache_timestamp is None:
            return False
        
        import datetime
        age = datetime.datetime.now() - self._data_cache_timestamp
        return age.total_seconds() < (max_age_minutes * 60)
    
    def clear_cache(self) -> None:
        """
        データキャッシュをクリア
        """
        self._cached_orders = None
        self._data_cache_timestamp = None
        self._prepared_data_cache = None
        logging.info("データキャッシュをクリアしました")
