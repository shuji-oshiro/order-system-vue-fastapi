"""
汎用モデル学習クラス

PyTorchモデルの学習プロセスを管理する汎用クラスです。
早期停止、損失追跡、バリデーション等の機能を提供します。
"""
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import DataLoader

from .callbacks import EarlyStopping, LossLogger
from .metrics import BinaryClassificationMetrics


class ModelTrainer:
    """汎用PyTorchモデル学習クラス"""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        # コールバック・メトリクス
        self.callbacks = []
        self.metrics = BinaryClassificationMetrics()
        
    def add_callback(self, callback):
        """コールバックを追加"""
        self.callbacks.append(callback)
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """1エポックの学習を実行"""
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_data in train_loader:
            batch_count += 1
            
            # データをデバイスに移動
            batch_data = [data.to(self.device) for data in batch_data]
            menu1_batch, menu2_batch, features_batch, targets_batch = batch_data
            
            # 勾配をゼロクリア
            self.optimizer.zero_grad()
            
            # フォワードプロパゲーション
            predictions = self.model.forward(menu1_batch, menu2_batch, features_batch)
            loss = self.criterion(predictions, targets_batch)
            
            # バックプロパゲーション
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # NaN/Infチェック
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"異常な損失値が発生しました: {loss.item()}")
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """バリデーションエポックを実行"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                # データをデバイスに移動
                batch_data = [data.to(self.device) for data in batch_data]
                menu1_batch, menu2_batch, features_batch, targets_batch = batch_data
                
                # 予測と損失計算
                predictions = self.model.forward(menu1_batch, menu2_batch, features_batch)
                loss = self.criterion(predictions, targets_batch)
                
                total_loss += loss.item()
                
                # メトリクス計算用にデータを収集
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets_batch.cpu().numpy())
        
        # メトリクス計算
        metrics_dict = self.metrics.calculate_metrics(all_targets, all_predictions)
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, metrics_dict
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        モデルを学習する
        
        Args:
            train_loader: 学習データローダー
            val_loader: バリデーションデータローダー
            epochs: 最大エポック数
            patience: 早期停止の許容エポック数
            
        Returns:
            学習結果の辞書
        """
        # 早期停止とロギングのコールバックを設定
        early_stopping = EarlyStopping(patience=patience)
        loss_logger = LossLogger()
        
        self.add_callback(early_stopping)
        self.add_callback(loss_logger)
        
        logging.info(f"学習開始: エポック数={epochs}, バッチサイズ={len(train_loader)}")
        
        for epoch in range(epochs):
            logging.info(f"=== エポック {epoch}/{epochs-1} 開始 ===")
            
            # 学習フェーズ
            train_loss = self.train_epoch(train_loader)
            
            # バリデーションフェーズ
            val_loss, metrics = self.validate_epoch(val_loader)
            
            # コールバック実行
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **metrics
                })
            
            logging.info(f"エポック {epoch}: 学習損失={train_loss:.4f}, バリデーション損失={val_loss:.4f}")
            logging.info(f"メトリクス: {metrics}")
            
            # 早期停止チェック
            if early_stopping.should_stop():
                logging.info(f"早期停止: エポック {epoch}")
                break
        
        # 学習結果を返す
        return {
            'train_losses': loss_logger.train_losses,
            'val_losses': loss_logger.val_losses,
            'final_train_loss': loss_logger.train_losses[-1] if loss_logger.train_losses else None,
            'final_val_loss': loss_logger.val_losses[-1] if loss_logger.val_losses else None,
            'best_val_loss': early_stopping.best_loss,
            'epochs_trained': len(loss_logger.train_losses),
            'early_stopped': early_stopping.should_stop()
        }
