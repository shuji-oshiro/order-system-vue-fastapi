from sqlalchemy.orm import Session
from backend.app.crud import order_crud
from backend.app.service.recommendtypes.recommend_strategy import RecommendStrategy

class Phase1FrequencyStrategy(RecommendStrategy):
    """Phase 1: 頻度ベースレコメンド"""
    
    def recommend(self, menu_id: int, db: Session) -> int:
        # 指定メニューと一緒に注文されたメニューの頻度を取得
        cooccurrence_results = order_crud.get_menu_cooccurrence_frequency(db, menu_id)
        
        if not cooccurrence_results:
            raise ValueError("共起するメニューが見つかりません")
        
        # 最も頻度の高いメニューを返す
        return cooccurrence_results[0][0]  # (menu_id, frequency) の最初の要素