"""
Machine Learning Module

リファクタリング後のML モジュール
モジュール別に機能を分離し、保守性とスケーラビリティを向上
"""

# Base classes
from .base.abstract_model import BaseRecommendModel
from .base.py_torch_base_model import PyTorchBaseModel

# Model implementations
from .models.neural_cf.model import NeuralCollaborativeFiltering

# Data processing
from .data.cache import DataCache
from .data.preprocessing import MenuDataPreprocessor

# Training
from .training.trainer import ModelTrainer
from .training.callbacks import EarlyStopping, LossLogger
from .training.metrics import BinaryClassificationMetrics

# Inference
from .inference.predictor import MenuRecommendationPredictor
from .inference.model_loader import ModelLoader
from .inference.batch_predictor import BatchPredictor

# Utils
from .utils.device_manager import DeviceManager
from .utils.model_utils import ModelUtils
from .utils.logging_config import setup_logging, get_ml_logger

__all__ = [
    # Base
    'BaseRecommendModel',
    'PyTorchBaseModel',
    
    # Models
    'NeuralCollaborativeFiltering',
    
    # Data
    'DataCache',
    'MenuDataPreprocessor',
    
    # Training  
    'ModelTrainer',
    'EarlyStopping',
    'LossLogger',
    'BinaryClassificationMetrics',
    
    # Inference
    'MenuRecommendationPredictor',
    'ModelLoader', 
    'BatchPredictor',
    
    # Utils
    'DeviceManager',
    'ModelUtils',
    'setup_logging',
    'get_ml_logger'
]
