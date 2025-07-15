import torch
import torch.nn as nn
from backend.app.ml.base.abstract_model import BaseRecommendModel

class PyTorchBaseModel(BaseRecommendModel, nn.Module):
    """PyTorchベースのレコメンドモデル"""
    
    def __init__(self, model_name: str):
        BaseRecommendModel.__init__(self, model_name)
        nn.Module.__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def save_model(self, path: str) -> None:
        """PyTorchモデルを保存する"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'is_trained': self.is_trained,
        }, path)
        
    def load_model(self, path: str) -> None:
        """PyTorchモデルを読み込む"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.model_name = checkpoint['model_name']
        self.is_trained = checkpoint['is_trained']
        self.to(self.device)
        
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """テンソルをデバイスに移動"""
        return tensor.to(self.device)