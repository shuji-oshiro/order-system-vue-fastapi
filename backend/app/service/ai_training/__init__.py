"""
AI Training Module

AI モデルのトレーニング機能を提供するモジュール
レコメンド機能とは完全に分離されている
"""

from .ai_model_trainer import AIModelTrainer
from .ai_training_api import router as training_router

__all__ = [
    "AIModelTrainer",
    "training_router"
]
