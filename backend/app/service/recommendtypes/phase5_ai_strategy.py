"""
Phase5: AI学習アルゴリズム戦略クラス

Neural Collaborative Filteringを使用したレコメンド戦略
学習済みモデルのみを使用（学習機能は分離済み）
"""
import logging
from typing import Dict, Any, List, Optional, Union
from sqlalchemy.orm import Session

from backend.app.crud import menu_crud
from backend.app.ml.models.neural_cf.model import NeuralCollaborativeFiltering
from backend.app.ml.inference.model_loader import ModelLoader
from backend.app.ml.utils.device_manager import DeviceManager
from backend.app.service.recommendtypes.phase4_complex_scoring_strategy import Phase4ComplexScoringStrategy
from backend.app.service.recommendtypes.recommend_strategy import RecommendStrategy

class Phase5AIRecommendStrategy(RecommendStrategy):
    """Phase 5: AI学習アルゴリズム（Neural Collaborative Filtering）"""
    
    def __init__(self):
        self.model = None
        self.model_loader = ModelLoader()
        self.device_manager = DeviceManager()
        self.model_name = "neural_cf"
        
    def _load_trained_model(self, db: Session) -> Optional[NeuralCollaborativeFiltering]:
        """
        学習済みモデルを読み込む（学習機能なし）
        
        Args:
            db: データベースセッション
            
        Returns:
            学習済みモデル（存在しない場合はNone）
        """
        if self.model is not None:
            return self.model
        
        try:
            # 学習済みモデルの存在確認
            available_models = self.model_loader.get_available_models(self.model_name)
            if "latest.pth" not in available_models:
                logging.warning("学習済みモデルが見つかりません")
                return None
            
            # デバイスを取得
            device = self.device_manager.get_device()
            
            # モデル初期化
            self.model = NeuralCollaborativeFiltering(db=db)
            
            # 学習済みモデルを読み込み
            logging.info(f"学習済みモデルを読み込み: {self.model_name}/latest.pth")
            loaded_model = self.model_loader.load_model(self.model, self.model_name, device=device)
            
            # 型エラー回避のため、明示的にキャスト
            if isinstance(loaded_model, NeuralCollaborativeFiltering):
                self.model = loaded_model
                logging.info("学習済みモデルの読み込みが完了しました")
                return self.model
            else:
                logging.error("モデルの型が不正です")
                return None
                
        except Exception as e:
            logging.error(f"モデル読み込みに失敗: {e}")
            return None
    
    def recommend(self, menu_id: int, db: Session) -> int:
        """
        AI学習アルゴリズムを使用してメニューを推薦
        
        Args:
            menu_id: 基準となるメニューID
            db: データベースセッション
            
        Returns:
            推薦するメニューID
        """
        try:
            # 学習済みモデルを読み込み
            model = self._load_trained_model(db)
            
            if model is None:
                # 学習済みモデルが存在しない場合はフォールバック
                logging.warning("学習済みモデルが存在しないため、Phase4にフォールバックします")
                return self._fallback_recommend(menu_id, db)
            
            # メニュー間関連性に基づく推薦の実行
            recommendations = model.recommend_menus(
                base_menu_id=menu_id,
                top_k=5,
                db=db,
                exclude_menu_ids=[menu_id]
            )
            
            if not recommendations:
                # 推薦結果が空の場合はフォールバック
                logging.warning("AI推薦結果が空のため、Phase4にフォールバックします")
                return self._fallback_recommend(menu_id, db)
            
            # 最高スコアのメニューを返す
            recommended_menu = recommendations[0]
            recommended_menu_id = recommended_menu['menu_id']
            score = recommended_menu['relationship_score']
            
            logging.info(f"AI推薦完了: メニューID {recommended_menu_id} (関連度スコア: {score:.4f})")
            
            return recommended_menu_id
            
        except Exception as e:
            logging.error(f"AI推薦でエラーが発生: {e}")
            return self._fallback_recommend(menu_id, db)
    
    def _fallback_recommend(self, menu_id: int, db: Session) -> int:
        """
        フォールバック推薦（Phase4に委譲）
        
        Args:
            menu_id: 基準となるメニューID
            db: データベースセッション
            
        Returns:
            推薦するメニューID
        """
        try:
            fallback_strategy = Phase4ComplexScoringStrategy()
            return fallback_strategy.recommend(menu_id, db)
        except Exception as e:
            logging.error(f"フォールバック推薦でもエラーが発生: {e}")
            # 最後の手段として、同じカテゴリの別のメニューを返す
            return self._emergency_recommend(menu_id, db)
    
    def _emergency_recommend(self, menu_id: int, db: Session) -> int:
        """
        緊急時推薦（同じカテゴリの別のメニュー）
        
        Args:
            menu_id: 基準となるメニューID
            db: データベースセッション
            
        Returns:
            推薦するメニューID
        """
        try:
            # 基準メニューを取得
            base_menu = menu_crud.get_menu_by_id(db, menu_id)
            if not base_menu:
                # 基準メニューが見つからない場合は最初のメニューを返す
                all_menus = menu_crud.get_all_menus(db)
                return all_menus[0].id if all_menus else menu_id
            
            # 同じカテゴリの別のメニューを取得
            same_category_menus = menu_crud.get_menus_by_category(db, base_menu.category_id)
            
            # 基準メニュー以外から選択
            other_menus = [m for m in same_category_menus if m.id != menu_id]
            
            if other_menus:
                recommended_id = other_menus[0].id
                logging.info(f"緊急時推薦: 同じカテゴリのメニューID {recommended_id}")
                return recommended_id
            else:
                # 同じカテゴリに他のメニューがない場合は基準メニューをそのまま返す
                logging.warning("推薦できるメニューが見つからないため、基準メニューを返します")
                return menu_id
                
        except Exception as e:
            logging.error(f"緊急時推薦でもエラーが発生: {e}")
            return menu_id
