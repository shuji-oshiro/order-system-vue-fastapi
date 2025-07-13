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
        dropout_rate: float = 0.2
    ):
        super().__init__("NeuralCollaborativeFiltering")
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
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
        
    def _init_weights(self):
        """重みの初期化"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
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
    
    def prepare_data(self, db: Session) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        データベースから学習データを準備
        
        Args:
            db: データベースセッション
            
        Returns:
            user_ids, item_ids, ratings (implicit feedback: 1 for interaction, 0 for no interaction)
        """
        logging.info("データの準備を開始...")
        
        # 注文履歴を取得
        orders = order_crud.get_all_orders(db)
        
        if not orders:
            raise ValueError("注文データが見つかりません")
        
        # DataFrameに変換
        order_data = []
        for order in orders:
            order_data.append({
                'seat_id': order.seat_id,
                'menu_id': order.menu_id,
                'quantity': order.quantity,
                'order_datetime': order.order_datetime
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
        
        # Negative sampling: 注文されていない組み合わせから同数サンプリング
        all_users = set(positive_interactions['user_encoded'].unique())
        all_items = set(positive_interactions['item_encoded'].unique())
        
        negative_interactions = []
        positive_set = set(zip(positive_interactions['user_encoded'], positive_interactions['item_encoded']))
        
        # ネガティブサンプリング（ポジティブサンプルの2倍）
        neg_samples_needed = len(positive_interactions) * 2
        neg_count = 0
        
        for user in all_users:
            for item in all_items:
                if (user, item) not in positive_set and neg_count < neg_samples_needed:
                    negative_interactions.append({'user_encoded': user, 'item_encoded': item, 'rating': 0})
                    neg_count += 1
                if neg_count >= neg_samples_needed:
                    break
            if neg_count >= neg_samples_needed:
                break
        
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
    
    def train(self, db: Session, **kwargs) -> Dict[str, Any]:
        """
        モデルを学習
        
        Args:
            db: データベースセッション
            **kwargs: 学習パラメータ
            
        Returns:
            学習結果
        """
        # パラメータ設定
        epochs = kwargs.get('epochs', 100)
        batch_size = kwargs.get('batch_size', 256)
        learning_rate = kwargs.get('learning_rate', 0.001)
        test_size = kwargs.get('test_size', 0.2)
        
        # データ準備
        user_ids, item_ids, ratings = self.prepare_data(db)
        
        # 訓練・テストデータ分割
        (user_train, user_test, 
         item_train, item_test, 
         rating_train, rating_test) = train_test_split(
            user_ids, item_ids, ratings, 
            test_size=test_size, 
            random_state=42, 
            stratify=ratings
        )
        
        # DataLoader
        train_dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(user_train),
            torch.LongTensor(item_train),
            torch.FloatTensor(rating_train)
        )
        
        test_dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(user_test),
            torch.LongTensor(item_test),
            torch.FloatTensor(rating_test)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        # 損失関数と最適化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # 学習ループ
        train_losses = []
        test_losses = []
        
        logging.info(f"学習開始: エポック数={epochs}, バッチサイズ={batch_size}")
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            
            for user_batch, item_batch, rating_batch in train_loader:
                user_batch = self.to_device(user_batch)
                item_batch = self.to_device(item_batch)
                rating_batch = self.to_device(rating_batch)
                
                optimizer.zero_grad()
                
                predictions = self.forward(user_batch, item_batch)
                loss = criterion(predictions, rating_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for user_batch, item_batch, rating_batch in test_loader:
                    user_batch = self.to_device(user_batch)
                    item_batch = self.to_device(item_batch)
                    rating_batch = self.to_device(rating_batch)
                    
                    predictions = self.forward(user_batch, item_batch)
                    loss = criterion(predictions, rating_batch)
                    
                    test_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)
            
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")
        
        self.is_trained = True
        logging.info("学習完了")
        
        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': train_losses[-1],
            'final_test_loss': test_losses[-1]
        }
    
    def predict(self, user_id: int, menu_id: int, **kwargs) -> float:
        """
        単一の予測を行う
        
        Args:
            user_id: 座席ID
            menu_id: メニューID
            
        Returns:
            推薦スコア (0-1)
        """
        if not self.is_trained:
            raise ValueError("モデルが学習されていません")
        
        try:
            # IDをエンコード
            user_encoded_array = self.user_encoder.transform([user_id])
            item_encoded_array = self.item_encoder.transform([menu_id])
            user_encoded = int(user_encoded_array[0])  # type: ignore
            item_encoded = int(item_encoded_array[0])  # type: ignore
            
            # テンソルに変換
            user_tensor = torch.LongTensor([user_encoded]).to(self.device)
            item_tensor = torch.LongTensor([item_encoded]).to(self.device)
            
            # 予測
            self.eval()
            with torch.no_grad():
                score = self.forward(user_tensor, item_tensor)
                return float(score.cpu().numpy())
        
        except (ValueError, KeyError):
            # 未知のIDの場合は平均スコアを返す
            return 0.5
    
    def recommend(self, user_id: int, exclude_menu_ids: Optional[List[int]] = None, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        ユーザーに対する推薦を行う
        
        Args:
            user_id: 座席ID
            exclude_menu_ids: 除外するメニューIDのリスト
            top_k: 推薦する件数
            
        Returns:
            (menu_id, score)のリスト（スコア降順）
        """
        if not self.is_trained:
            raise ValueError("モデルが学習されていません")
        
        if exclude_menu_ids is None:
            exclude_menu_ids = []
        
        # 全メニューに対するスコアを計算
        all_items = self.item_encoder.classes_
        recommendations = []
        
        for menu_id in all_items:
            if menu_id not in exclude_menu_ids:
                score = self.predict(user_id, menu_id)
                recommendations.append((int(menu_id), float(score)))
        
        # スコア降順でソート
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:top_k]
