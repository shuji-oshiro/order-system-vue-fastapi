# ML Base Classes
"""
ML基底クラス・共通機能モジュール

このモジュールには、全てのMLモデルが継承する基底クラスと
共通的に使用される機能が含まれています。
"""

from .abstract_model import BaseRecommendModel
from .py_torch_base_model import PyTorchBaseModel

__all__ = [
    'BaseRecommendModel',
    'PyTorchBaseModel',
]
