"""
推論実行クラス

学習済みモデルを使ったメニュー推薦の推論を実行します。
"""
import torch
import logging
import numpy as np
from typing import cast
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from backend.app.models.model import Menu
from backend.app.ml.data.cache import DataCache
from backend.app.ml.data.preprocessing import MenuDataPreprocessor


class MenuRecommendationPredictor:
    """メニュー推薦推論クラス"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.data_cache = DataCache()
        self.preprocessor = MenuDataPreprocessor()
        
        # モデルを評価モードに設定
        self.model.eval()
        
    def predict_menu_relationships(
        self, 
        base_menu_id: int, 
        candidate_menu_ids: List[int],
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        基準メニューに対する候補メニューとの関連度を予測
        
        Args:
            base_menu_id: 基準となるメニューID
            candidate_menu_ids: 推薦候補のメニューIDリスト
            db: データベースセッション
            
        Returns:
            関連度スコア付きのメニューリスト
        """
        logging.info(f"メニュー関連度予測開始: 基準={base_menu_id}, 候補={len(candidate_menu_ids)}件")
        
        # 特徴量を準備
        features_data = self._prepare_prediction_features(
            base_menu_id, candidate_menu_ids, db
        )
        
        predictions = []
        
        with torch.no_grad():
            for candidate_id, features in features_data.items():
                # テンソルに変換
                menu1_tensor = torch.LongTensor([features['menu1_encoded']]).to(self.device)
                menu2_tensor = torch.LongTensor([features['menu2_encoded']]).to(self.device)
                features_tensor = torch.FloatTensor([features['features']]).to(self.device)
                
                # 予測実行
                score = self.model.forward(menu1_tensor, menu2_tensor, features_tensor)
                
                predictions.append({
                    'menu_id': candidate_id,
                    'relationship_score': float(score.item()),
                    'freq_similarity': features['features'][0],
                    'time_similarity': features['features'][1],
                    'category_similarity': features['features'][2]
                })
        
        # スコア順でソート
        predictions.sort(key=lambda x: x['relationship_score'], reverse=True)
        
        logging.info(f"予測完了: トップスコア={predictions[0]['relationship_score']:.4f}")
        return predictions
    
    def recommend_menus(
        self, 
        base_menu_id: int, 
        top_k: int = 5,
        db: Optional[Session] = None,
        exclude_menu_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        基準メニューに対する推薦メニューを取得
        
        Args:
            base_menu_id: 基準となるメニューID
            top_k: 推薦するメニュー数
            db: データベースセッション
            exclude_menu_ids: 除外するメニューIDのリスト
            
        Returns:
            推薦メニューのリスト
        """
        # 全メニューを候補として取得
        if db is None:
            logging.error("データベースセッションが提供されていません")
            return []
        all_menus = db.query(Menu).all()
        candidate_ids = [menu.id for menu in all_menus if menu.id != base_menu_id]
        
        # if db is not None:
            
        #     all_menus = db.query(Menu).all()
        #     candidate_ids = [menu.id for menu in all_menus if menu.id != base_menu_id]
        # else:
        #     # キャッシュから取得（簡易版）
        #     orders = self.data_cache.get_cached_orders()
        #     if orders:
        #         candidate_ids = list(set(order.menu_id for order in orders if order.menu_id != base_menu_id))
        #     else:
        #         raise ValueError("推薦に必要なデータが不足しています")
        
        # 除外メニューを削除
        if exclude_menu_ids:
            candidate_ids = [mid for mid in candidate_ids if mid not in exclude_menu_ids]
        
        # 関連度を予測
        predictions = self.predict_menu_relationships(base_menu_id, candidate_ids, db)
        
        # 上位K件を返す
        return predictions[:top_k]
    
    def _prepare_prediction_features(
        self, 
        base_menu_id: int, 
        candidate_menu_ids: List[int],
        db: Optional[Session] = None
    ) -> Dict[int, Dict[str, Any]]:
        """推論用の特徴量を準備"""
        
        # エンコーダーが初期化されていない場合、学習データで初期化
        if not hasattr(self.preprocessor.menu_encoder, 'classes_'):
            logging.info("メニューエンコーダーを初期化します")
            self._initialize_encoder(db)
        
        # 基準メニューのエンコード値を取得
        base_encoded = self._safe_encode_menu(base_menu_id)
        
        # 候補メニューの特徴量を準備
        features_data = {}
        
        for candidate_id in candidate_menu_ids:
            candidate_encoded = self._safe_encode_menu(candidate_id)
            
            # デフォルト特徴量（実際の計算は後で改善）
            default_features = [0.5, 0.5, 0.0]  # freq_sim, time_sim, category_sim
            
            features_data[candidate_id] = {
                'menu1_encoded': base_encoded,
                'menu2_encoded': candidate_encoded,
                'features': default_features
            }
        
        return features_data
    
    def _safe_encode_menu(self, menu_id: int) -> int:
        """メニューIDを安全にエンコード（未知IDは最大値+1を返す）"""
        try:
            result = self.preprocessor.menu_encoder.transform([menu_id])
            return cast(np.ndarray, result)[0]
        except ValueError:
            # 未知のメニューID: 学習済みメニュー数を返す（新しいインデックス）
            logging.warning(f"未知のメニューID {menu_id} を検出、新しいエンコード値を割り当て")
            return len(self.preprocessor.menu_encoder.classes_)
    
    def _initialize_encoder(self, db: Optional[Session] = None):
        """エンコーダーを初期化（学習データから）"""
        if db is None:
            raise ValueError("エンコーダー初期化にはデータベースセッションが必要です")
        
        # 学習時と同じ方法でメニューエンコーダーを初期化
        self.preprocessor.prepare_menu_pairs(db=db)
