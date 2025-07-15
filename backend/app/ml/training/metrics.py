"""
評価メトリクス

機械学習モデルの性能評価指標を計算します。
"""
import numpy as np
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class BinaryClassificationMetrics:
    """二値分類メトリクス"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        
    def calculate_metrics(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """
        二値分類の各種メトリクスを計算
        
        Args:
            y_true: 実際のラベル
            y_pred: 予測確率
            
        Returns:
            メトリクスの辞書
        """
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        
        # 確率を二値ラベルに変換
        y_pred_binary = (y_pred_np >= self.threshold).astype(int)
        
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true_np, y_pred_binary)
            metrics['precision'] = precision_score(y_true_np, y_pred_binary, zero_division=0)
            metrics['recall'] = recall_score(y_true_np, y_pred_binary, zero_division=0)
            metrics['f1'] = f1_score(y_true_np, y_pred_binary, zero_division=0)
            
            # AUCは確率値で計算
            if len(np.unique(y_true_np)) > 1:  # 両方のクラスが存在する場合のみ
                metrics['auc'] = roc_auc_score(y_true_np, y_pred_np)
            else:
                metrics['auc'] = 0.0
                
        except Exception as e:
            # メトリクス計算でエラーが発生した場合はデフォルト値
            metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0
            }
            
        return metrics


class RegressionMetrics:
    """回帰メトリクス"""
    
    def calculate_metrics(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """
        回帰の各種メトリクスを計算
        
        Args:
            y_true: 実際の値
            y_pred: 予測値
            
        Returns:
            メトリクスの辞書
        """
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        
        # MSE (Mean Squared Error)
        mse = np.mean((y_true_np - y_pred_np) ** 2)
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(y_true_np - y_pred_np))
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)
        
        # R²スコア
        y_mean = np.mean(y_true_np)
        ss_res = np.sum((y_true_np - y_pred_np) ** 2)
        ss_tot = np.sum((y_true_np - y_mean) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }
