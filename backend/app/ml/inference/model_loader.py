"""
モデル読み込みクラス

学習済みモデルの読み込み・保存機能を提供します。
"""
import os
import torch
import logging
import dotenv
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any

dotenv.load_dotenv()
AI_MODEL_STORAGE_DIR = os.getenv("AI_MODEL_STORAGE_DIR", "backend/app/ml/saved_models")

class ModelLoader:
    """モデル読み込み・保存クラス"""
       
    def __init__(self):
        self.model_dir = Path(AI_MODEL_STORAGE_DIR)

    def save_model(
        self, 
        model: nn.Module, 
        model_name: str, 
        epoch: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        モデルを保存
        
        Args:
            model: 保存するモデル
            model_name: モデル名
            epoch: エポック数（指定した場合はファイル名に含める）
            metadata: 追加のメタデータ
            
        Returns:
            保存先パス
        """
        # 保存ディレクトリを作成
        save_dir = self.model_dir / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名を決定
        if epoch is not None:
            filename = f"epoch_{epoch}.pth"
        else:
            filename = "latest.pth"
        
        filepath = save_dir / filename
        
        # 保存データを準備
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_config': getattr(model, 'config', {}),
            'metadata': metadata or {}
        }
        
        # モデルを保存
        torch.save(save_data, filepath)
        logging.info(f"モデル保存完了: {filepath}")
        
        # latest.pthとしてもコピー保存
        if epoch is not None:
            latest_path = save_dir / "latest.pth"
            torch.save(save_data, latest_path)
        
        return str(filepath)
    
    def load_model(
        self, 
        model: nn.Module, 
        model_name: str, 
        filename: str = "latest.pth",
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """
        モデルを読み込み
        
        Args:
            model: 読み込み先のモデルインスタンス
            model_name: モデル名
            filename: 読み込むファイル名
            device: デバイス
            
        Returns:
            読み込まれたモデル
        """
        filepath = self.model_dir / model_name / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {filepath}")
        
        # デバイスを決定
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # モデルを読み込み
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logging.info(f"モデル読み込み完了: {filepath}")
        
        return model
    
    def get_available_models(self, model_name: str) -> list:
        """利用可能なモデルファイルのリストを取得"""
        model_dir = self.model_dir / model_name
        
        if not model_dir.exists():
            return []
        
        return [f.name for f in model_dir.glob("*.pth")]
    
    def get_model_metadata(self, model_name: str, filename: str = "latest.pth") -> Dict[str, Any]:
        """モデルのメタデータを取得"""
        filepath = self.model_dir / model_name / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        return {
            'model_class': checkpoint.get('model_class', 'Unknown'),
            'model_config': checkpoint.get('model_config', {}),
            'metadata': checkpoint.get('metadata', {}),
            'file_size_mb': filepath.stat().st_size / (1024 * 1024)
        }
