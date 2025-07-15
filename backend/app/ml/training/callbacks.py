"""
学習コールバック

早期停止、損失ロギング等の学習プロセス制御機能を提供します。
"""
import logging
from typing import Dict, Any, List


class BaseCallback:
    """基底コールバッククラス"""
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """エポック終了時に呼び出される"""
        pass


class EarlyStopping(BaseCallback):
    """早期停止コールバック"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped = False
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """バリデーション損失をチェックして早期停止を判定"""
        current_loss = logs.get('val_loss', float('inf'))
        
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped = True
            logging.info(f"早期停止: {self.patience}エポック改善なし")
    
    def should_stop(self) -> bool:
        """停止すべきかどうかを返す"""
        return self.stopped


class LossLogger(BaseCallback):
    """損失記録コールバック"""
    
    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """損失を記録"""
        if 'train_loss' in logs:
            self.train_losses.append(logs['train_loss'])
        if 'val_loss' in logs:
            self.val_losses.append(logs['val_loss'])


class ModelCheckpoint(BaseCallback):
    """モデル保存コールバック"""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', save_best_only: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_value = float('inf') if 'loss' in monitor else float('-inf')
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """モデルを保存"""
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
            
        is_improvement = (
            (self.monitor.endswith('loss') and current_value < self.best_value) or
            (not self.monitor.endswith('loss') and current_value > self.best_value)
        )
        
        if not self.save_best_only or is_improvement:
            # TODO: モデル保存ロジックを実装
            logging.info(f"モデル保存: エポック {epoch}, {self.monitor}={current_value:.4f}")
            self.best_value = current_value
