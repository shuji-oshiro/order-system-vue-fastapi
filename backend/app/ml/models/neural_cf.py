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
        num_users: int, 
        num_items: int,
        embedding_dim: int = 50,
        hidden_dims: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        db: Optional[Session] = None  # DBセッションを初期化時に受け取る
    ):
        super().__init__("NeuralCollaborativeFiltering")
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # データキャッシュ用の変数（インスタンス変数として保持）
        self._cached_orders = None
        self._data_cache_timestamp = None
        self._prepared_data_cache = None
        
        # User and Item Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Neural MF layers
        layers = []
        input_dim = embedding_dim * 2  # user + item embeddings concatenated
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize embeddings
        self._init_weights()
        
        
        # Move to device
        self.to(self.device)
        
        # Label encoders for user and item IDs
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # 初期化時にデータを読み込む（オプショナル）
        if db is not None:
            self._load_and_cache_data(db)
        
    def _init_weights(self):
        """重みの初期化 - Xavier/He初期化を適用"""
        # Embeddings: 正規分布での初期化
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # MLPの重み初期化
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                # Xavier uniform初期化
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            user_ids: ユーザーIDのテンソル
            item_ids: アイテムIDのテンソル
            
        Returns:
            推薦スコア（0-1）
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        concat_emb = torch.cat([user_emb, item_emb], dim=1)
        
        # Pass through MLP
        output = self.mlp(concat_emb)
        
        return output.squeeze()
    
    def prepare_data(self, db: Optional[Session] = None, force_reload: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        データベースから学習データを準備（キャッシュ機能付き）
        
        Args:
            db: データベースセッション（Noneの場合はキャッシュを使用）
            force_reload: 強制的にデータを再読み込みするかどうか
            
        Returns:
            user_ids, item_ids, ratings (implicit feedback: 1 for interaction, 0 for no interaction)
        """
        logging.info("データの準備を開始...")
        
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
        
        # DataFrameに変換
        order_data = []
        for order in orders:
            order_data.append({
                'seat_id': order.seat_id,
                'menu_id': order.menu_id, 
                'quantity': order.order_cnt, # 注文数
                'order_datetime': order.order_date # 注文日時
            })
        
        df = pd.DataFrame(order_data)
        
        # 座席ID（ユーザー）とメニューID（アイテム）のエンコーディング
        unique_users = df['seat_id'].unique()
        unique_items = df['menu_id'].unique()
        
        self.user_encoder.fit(unique_users)
        self.item_encoder.fit(unique_items)
        
        # Positive interactions (実際の注文)
        df['user_encoded'] = self.user_encoder.transform(df['seat_id'])
        df['item_encoded'] = self.item_encoder.transform(df['menu_id'])
        
        # Implicit feedback: 注文されたら1
        positive_interactions = df[['user_encoded', 'item_encoded']].drop_duplicates()
        positive_interactions['rating'] = 1
        
        # Negative sampling: より効率的な実装
        import random
        
        all_users = set(positive_interactions['user_encoded'].unique())
        all_items = set(positive_interactions['item_encoded'].unique())
        
        negative_interactions = []
        positive_set = set(zip(positive_interactions['user_encoded'], positive_interactions['item_encoded']))
        
        # ネガティブサンプリング（ポジティブサンプルの2倍）
        neg_samples_needed = len(positive_interactions) * 2
        users_list = list(all_users)
        items_list = list(all_items)
        
        # ランダムサンプリングで効率化
        attempts = 0
        max_attempts = neg_samples_needed * 5  # 無限ループ回避
        
        while len(negative_interactions) < neg_samples_needed and attempts < max_attempts:
            user = random.choice(users_list)
            item = random.choice(items_list)
            
            if (user, item) not in positive_set:
                negative_interactions.append({
                    'user_encoded': user, 
                    'item_encoded': item, 
                    'rating': 0
                })
            
            attempts += 1
        
        negative_df = pd.DataFrame(negative_interactions)
        
        # Combine positive and negative samples
        final_data = pd.concat([positive_interactions, negative_df], ignore_index=True)
        final_data = final_data.sample(frac=1).reset_index(drop=True)  # Shuffle
        
        logging.info(f"準備完了: ポジティブサンプル {len(positive_interactions)}件, ネガティブサンプル {len(negative_interactions)}件")
        
        return (
            final_data['user_encoded'].values.astype(np.int64),
            final_data['item_encoded'].values.astype(np.int64),
            final_data['rating'].values.astype(np.float32)
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
        # 注文履歴からユーザー（座席）-アイテム（メニュー）の相互作用データを生成
        # キャッシュ機能により初回以降は高速化
        user_ids, item_ids, ratings = self.prepare_data(db, force_reload=force_reload)
        
        # ====================
        # 3. 訓練・テストデータの分割
        # ====================
        # 層化分割で0/1ラベルのバランスを保持しながらデータを分割
        (user_train, user_test, 
         item_train, item_test, 
         rating_train, rating_test) = train_test_split(
            user_ids, item_ids, ratings, 
            test_size=test_size,        # 20%をテストデータに
            random_state=42,            # 再現性のためのシード固定
            stratify=ratings            # 0/1ラベルの比率を保持
        )
        
        # ====================
        # 4. PyTorch DataLoaderの作成
        # ====================
        # NumPy配列からPyTorchテンソルに変換してDatasetを作成
        train_dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(user_train),    # ユーザーID（整数型）
            torch.LongTensor(item_train),    # アイテムID（整数型）
            torch.FloatTensor(rating_train)  # レーティング（0 or 1の浮動小数点型）
        )
        
        test_dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(user_test),
            torch.LongTensor(item_test),
            torch.FloatTensor(rating_test)
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
            for user_batch, item_batch, rating_batch in train_loader:
                batch_count += 1
                
                # 最初のエポックは詳細ログ、その後は間隔を空ける
                if epoch == 0 or batch_count % 20 == 0:
                    logging.info(f"  エポック {epoch}, バッチ {batch_count}/{len(train_loader)} 処理中...")
                
                # データの形状チェック（最初のバッチのみ）
                if epoch == 0 and batch_count == 1:
                    logging.info(f"  バッチサイズ: user={user_batch.shape}, item={item_batch.shape}, rating={rating_batch.shape}")
                
                # GPU/CPUにデータを移動
                user_batch = self.to_device(user_batch)
                item_batch = self.to_device(item_batch)
                rating_batch = self.to_device(rating_batch)
                
                # 勾配をゼロクリア（前のバッチの勾配をリセット）
                optimizer.zero_grad()
                
                # フォワードプロパゲーション（予測）
                predictions = self.forward(user_batch, item_batch)
                # 損失計算（予測値と実際値の差）
                loss = criterion(predictions, rating_batch)
                
                # デバッグ情報（最初のバッチのみ）
                if batch_count == 1:
                    logging.info(f"最初のバッチ - 予測値範囲: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
                    logging.info(f"最初のバッチ - 実際値範囲: [{rating_batch.min().item():.4f}, {rating_batch.max().item():.4f}]")
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
                for user_batch, item_batch, rating_batch in test_loader:
                    # GPU/CPUにデータを移動
                    user_batch = self.to_device(user_batch)
                    item_batch = self.to_device(item_batch)
                    rating_batch = self.to_device(rating_batch)
                    
                    # 予測と損失計算（勾配計算なし）
                    predictions = self.forward(user_batch, item_batch)
                    loss = criterion(predictions, rating_batch)
                    
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
            for user_batch, item_batch, rating_batch in test_loader:
                # データをデバイスに移動
                user_batch = self.to_device(user_batch)
                item_batch = self.to_device(item_batch)
                rating_batch = self.to_device(rating_batch)
                
                # 予測実行
                predictions = self.forward(user_batch, item_batch)
                # 損失計算
                loss = criterion(predictions, rating_batch)
                total_loss += loss.item()
                
                # Binary accuracy計算（閾値0.5で二値分類）
                binary_predictions = (predictions > 0.5).float()  # 0.5以上なら1、未満なら0
                correct_predictions += (binary_predictions == rating_batch).sum().item()
                total_predictions += rating_batch.size(0)
        
        # 精度計算（正解数 ÷ 総数）
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'test_accuracy': accuracy,                    # テスト精度（0-1）
            'test_loss_final': total_loss / len(test_loader)  # 平均テスト損失
        }
    
    def predict(self, user_id: int, menu_id: int, **kwargs) -> float:
        """
        単一の予測を行う
        
        Args:
            user_id: 座席ID
            menu_id: メニューID
            **kwargs: 予測パラメータ（将来の拡張用）
            
        Returns:
            推薦スコア (0-1)
        """
        # 学習済み状態の確認
        if not self.is_trained:
            raise ValueError("モデルが学習されていません。train()メソッドを先に実行してください。")
        
        # エンコーダーの学習状態確認
        if not hasattr(self.user_encoder, 'classes_') or not hasattr(self.item_encoder, 'classes_'):
            raise ValueError("エンコーダーが学習されていません。prepare_data()が正しく実行されていない可能性があります。")
        
        try:
            # IDをエンコード（エラーハンドリング強化）
            user_encoded_array = self.user_encoder.transform([user_id])
            item_encoded_array = self.item_encoder.transform([menu_id])
            user_encoded = int(user_encoded_array[0])  # type: ignore
            item_encoded = int(item_encoded_array[0])  # type: ignore
            
            # テンソルに変換
            user_tensor = torch.LongTensor([user_encoded]).to(self.device)
            item_tensor = torch.LongTensor([item_encoded]).to(self.device)
            
            # 予測実行（評価モードで確実に実行）
            self.eval()  # 評価モードに設定（Dropout無効化）
            with torch.no_grad():
                score = self.forward(user_tensor, item_tensor)
                return float(score.cpu().numpy())
        
        except (ValueError, KeyError) as e:
            # 未知のIDの場合は詳細なログと平均スコアを返す
            logging.warning(f"未知のID detected - user_id: {user_id}, menu_id: {menu_id}, error: {str(e)}")
            return 0.5
    
    def recommend(self, user_id: int, exclude_menu_ids: Optional[List[int]] = None, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        ユーザーに対する推薦を行う
        
        Args:
            user_id: 座席ID
            exclude_menu_ids: 除外するメニューIDのリスト（デフォルト: None）
            top_k: 推薦する件数（デフォルト: 10）
            
        Returns:
            (menu_id, score)のリスト（スコア降順）
        """
        # 学習済み状態の確認
        if not self.is_trained:
            raise ValueError("モデルが学習されていません。train()メソッドを先に実行してください。")
        
        # エンコーダーの学習状態確認
        if not hasattr(self.item_encoder, 'classes_'):
            raise ValueError("アイテムエンコーダーが学習されていません。prepare_data()が正しく実行されていない可能性があります。")
        
        if exclude_menu_ids is None:
            exclude_menu_ids = []
        
        # 全メニューに対するスコアを計算
        all_items = self.item_encoder.classes_
        recommendations = []
        
        logging.info(f"推薦計算開始 - user_id: {user_id}, 対象メニュー数: {len(all_items)}")
        
        for menu_id in all_items:
            if menu_id not in exclude_menu_ids:
                try:
                    score = self.predict(user_id, menu_id)
                    recommendations.append((int(menu_id), float(score)))
                except Exception as e:
                    # 個別のメニューで予測エラーが発生した場合はスキップ
                    logging.warning(f"メニューID {menu_id} の予測でエラー: {str(e)}")
                    continue
        
        # スコア降順でソート
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        logging.info(f"推薦計算完了 - 推薦候補数: {len(recommendations)}, 返却数: {min(top_k, len(recommendations))}")
        
        return recommendations[:top_k]
    
    def _validate_encoders(self) -> bool:
        """
        エンコーダーの学習状態を検証
        
        Returns:
            bool: エンコーダーが正しく学習されている場合True
        """
        try:
            # user_encoderの状態確認
            if not hasattr(self.user_encoder, 'classes_'):
                logging.error("user_encoderが学習されていません")
                return False
            
            # item_encoderの状態確認    
            if not hasattr(self.item_encoder, 'classes_'):
                logging.error("item_encoderが学習されていません")
                return False
            
            # クラス数とモデルの次元数の整合性確認
            if len(self.user_encoder.classes_) > self.num_users:
                logging.warning(f"user_encoder classes ({len(self.user_encoder.classes_)}) > num_users ({self.num_users})")
                
            if len(self.item_encoder.classes_) > self.num_items:
                logging.warning(f"item_encoder classes ({len(self.item_encoder.classes_)}) > num_items ({self.num_items})")
            
            logging.info(f"エンコーダー検証完了 - Users: {len(self.user_encoder.classes_)}, Items: {len(self.item_encoder.classes_)}")
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
            'num_users': self.num_users,
            'num_items': self.num_items
        }
        
        # エンコーダー情報
        if hasattr(self.user_encoder, 'classes_'):
            info['encoded_users'] = len(self.user_encoder.classes_)
            info['user_ids_range'] = f"{self.user_encoder.classes_.min()}-{self.user_encoder.classes_.max()}"
        else:
            info['encoded_users'] = 0
            
        if hasattr(self.item_encoder, 'classes_'):
            info['encoded_items'] = len(self.item_encoder.classes_)
            info['item_ids_range'] = f"{self.item_encoder.classes_.min()}-{self.item_encoder.classes_.max()}"
        else:
            info['encoded_items'] = 0
            
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
