"""
ML レコメンドモデルのベースクラス
"""
from sqlalchemy.orm import Session
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional


class BaseRecommendModel(ABC):
    """レコメンドモデルのベースクラス"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
        
    @abstractmethod
    def fit(self, db: Session, **kwargs) -> Dict[str, Any]:
        """
        モデルを学習する
        
        Args:
            db: データベースセッション
            **kwargs: 学習パラメータ
            
        Returns:
            Dict[str, Any]: 学習結果（損失、精度等）
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, menu_id: int, **kwargs) -> float:
        """
        予測を行う
        
        Args:
            user_id: ユーザー（座席）ID
            menu_id: メニューID
            **kwargs: 予測時の追加パラメータ
            
        Returns:
            float: 推薦スコア
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, exclude_menu_ids: Optional[List[int]] = None, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        ユーザーに対する推薦を行う
        
        Args:
            user_id: ユーザー（座席）ID
            exclude_menu_ids: 除外するメニューIDのリスト
            top_k: 推薦する件数
            
        Returns:
            List[Tuple[int, float]]: (menu_id, score)のリスト
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """モデルを保存する"""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """モデルを読み込む"""
        pass

