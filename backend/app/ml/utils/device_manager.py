"""
デバイス管理ユーティリティ

GPU/CPU の選択と管理を行う機能を提供します。
"""
import torch
import logging
from typing import Optional


class DeviceManager:
    """GPU/CPUデバイス管理クラス"""
    
    def __init__(self):
        self._device = None
        self._cuda_available = torch.cuda.is_available()
        
    def get_device(self, prefer_gpu: bool = True) -> torch.device:
        """
        最適なデバイスを取得
        
        Args:
            prefer_gpu: GPUを優先するかどうか
            
        Returns:
            PyTorchデバイス
        """
        if self._device is not None:
            return self._device
            
        if prefer_gpu and self._cuda_available:
            self._device = torch.device('cuda')
            logging.info(f"GPU使用: {torch.cuda.get_device_name()}")
        else:
            self._device = torch.device('cpu')
            logging.info("CPU使用")
            
        return self._device
    
    def set_device(self, device: str) -> torch.device:
        """
        手動でデバイスを設定
        
        Args:
            device: デバイス名 ('cuda', 'cpu', 'cuda:0' など)
            
        Returns:
            設定されたデバイス
        """
        try:
            self._device = torch.device(device)
            logging.info(f"デバイス設定: {self._device}")
            return self._device
        except Exception as e:
            logging.warning(f"デバイス設定失敗: {e}, CPUを使用")
            self._device = torch.device('cpu')
            return self._device
    
    def get_memory_info(self) -> dict:
        """
        GPU メモリ情報を取得
        
        Returns:
            メモリ情報の辞書
        """
        if not self._cuda_available:
            return {'error': 'CUDA not available'}
            
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            return {
                'allocated_gb': round(allocated, 2),
                'reserved_gb': round(reserved, 2),
                'total_gb': round(total, 2),
                'free_gb': round(total - reserved, 2)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def clear_cache(self):
        """GPU キャッシュをクリア"""
        if self._cuda_available:
            torch.cuda.empty_cache()
            logging.info("GPU キャッシュをクリアしました")
    
    @property
    def cuda_available(self) -> bool:
        """CUDA が利用可能かどうか"""
        return self._cuda_available
    
    @property
    def current_device(self) -> Optional[torch.device]:
        """現在のデバイス"""
        return self._device
