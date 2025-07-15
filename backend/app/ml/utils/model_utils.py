"""
モデル関連ユーティリティ

PyTorchモデルの操作、保存、読み込み等の汎用的な機能を提供します。
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


class ModelUtils:
    """モデル関連ユーティリティクラス"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        モデルのパラメータ数を計算
        
        Args:
            model: PyTorchモデル
            
        Returns:
            パラメータ数の詳細
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
    
    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """
        モデルのメモリサイズを計算（MB）
        
        Args:
            model: PyTorchモデル
            
        Returns:
            モデルサイズ（MB）
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return round(size_mb, 2)
    
    @staticmethod
    def freeze_layers(model: nn.Module, layer_names: List[str]) -> nn.Module:
        """
        指定したレイヤーの重みを凍結
        
        Args:
            model: PyTorchモデル
            layer_names: 凍結するレイヤー名のリスト（Noneの場合は全レイヤー）
            
        Returns:
            更新されたモデル
        """
        if layer_names is None:
            # 全パラメータを凍結
            for param in model.parameters():
                param.requires_grad = False
            logging.info("全レイヤーを凍結しました")
        else:
            # 指定されたレイヤーのみ凍結
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = False
                    logging.info(f"レイヤー凍結: {name}")
        
        return model
    
    @staticmethod
    def unfreeze_layers(model: nn.Module, layer_names: List[str]) -> nn.Module:
        """
        指定したレイヤーの重みを解凍
        
        Args:
            model: PyTorchモデル
            layer_names: 解凍するレイヤー名のリスト（Noneの場合は全レイヤー）
            
        Returns:
            更新されたモデル
        """
        if layer_names is None:
            # 全パラメータを解凍
            for param in model.parameters():
                param.requires_grad = True
            logging.info("全レイヤーを解凍しました")
        else:
            # 指定されたレイヤーのみ解凍
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
                    logging.info(f"レイヤー解凍: {name}")
        
        return model
    
    @staticmethod
    def get_layer_names(model: nn.Module) -> List[str]:
        """
        モデルのレイヤー名一覧を取得
        
        Args:
            model: PyTorchモデル
            
        Returns:
            レイヤー名のリスト
        """
        return [name for name, _ in model.named_modules()]
    
    @staticmethod
    def print_model_summary(model: nn.Module, input_size: Tuple[int, ...]):
        """
        モデルの詳細サマリーを出力
        
        Args:
            model: PyTorchモデル
            input_size: 入力サイズ（指定した場合はforward計算も実行）
        """
        print("=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        print(f"Model: {model.__class__.__name__}")
        
        # パラメータ数
        param_info = ModelUtils.count_parameters(model)
        print(f"Total parameters: {param_info['total_parameters']:,}")
        print(f"Trainable parameters: {param_info['trainable_parameters']:,}")
        print(f"Non-trainable parameters: {param_info['non_trainable_parameters']:,}")
        
        # モデルサイズ
        model_size = ModelUtils.get_model_size_mb(model)
        print(f"Model size: {model_size} MB")
        
        # レイヤー構造
        print("\nLayers:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 末端のレイヤーのみ表示
                print(f"  {name}: {module}")
        
        print("=" * 60)
    
    @staticmethod
    def save_model_checkpoint(
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        epoch: int, 
        loss: float, 
        filepath: str,
        metadata: Dict[str, Any]
    ):
        """
        モデルチェックポイントを保存
        
        Args:
            model: PyTorchモデル
            optimizer: 最適化器
            epoch: エポック数
            loss: 損失値
            filepath: 保存先パス
            metadata: 追加のメタデータ
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metadata': metadata or {}
        }
        
        # ディレクトリを作成
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, filepath)
        logging.info(f"チェックポイント保存: {filepath}")
    
    @staticmethod
    def load_model_checkpoint(
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        filepath: str,
        device: torch.device
    ) -> Dict[str, Any]:
        """
        モデルチェックポイントを読み込み
        
        Args:
            model: PyTorchモデル
            optimizer: 最適化器
            filepath: 読み込み先パス
            device: デバイス
            
        Returns:
            チェックポイント情報
        """
        if device is None:
            device = torch.device('cpu')
            
        checkpoint = torch.load(filepath, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logging.info(f"チェックポイント読み込み: {filepath}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', 0.0),
            'metadata': checkpoint.get('metadata', {})
        }
