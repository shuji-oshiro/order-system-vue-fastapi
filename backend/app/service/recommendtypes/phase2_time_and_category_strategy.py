from sqlalchemy.orm import Session
from backend.app.crud import order_crud
from backend.app.models.model import Menu
from backend.app.service.recommendtypes.recommend_strategy import RecommendStrategy
from backend.app.service.recommendtypes.phase1_frequency_strategy import Phase1FrequencyStrategy

class Phase2TimeAndCategoryStrategy(RecommendStrategy):
    """Phase 2: 時間帯 + カテゴリ親和性レコメンド"""
    
    def recommend(self, menu_id: int, db: Session) -> int:
        from datetime import datetime
        
        # 基準メニューの情報を取得
        base_menu = db.query(Menu).filter(Menu.id == menu_id).first()
        if not base_menu:
            raise ValueError("基準メニューが見つかりません")
        
        candidate_scores = {}
        
        # 1. 頻度ベーススコア（重み: 50%）
        try:
            cooccurrence_results = order_crud.get_menu_cooccurrence_frequency(db, menu_id)
            max_freq = max([result[1] for result in cooccurrence_results]) if cooccurrence_results else 1
            
            for candidate_menu_id, frequency in cooccurrence_results:
                # 基準メニュー自身は除外
                if candidate_menu_id != menu_id:
                    if candidate_menu_id not in candidate_scores:
                        candidate_scores[candidate_menu_id] = 0
                    candidate_scores[candidate_menu_id] += (frequency / max_freq) * 0.5
        except:
            pass
        
        # 2. 時間帯スコア（重み: 30%）
        try:
            current_hour = datetime.now().hour
            popular_menus = order_crud.get_popular_menus_by_time_period(db, current_hour)
            max_time_freq = max([result[1] for result in popular_menus]) if popular_menus else 1
            
            for candidate_menu_id, frequency in popular_menus:
                # 基準メニュー自身は除外
                if candidate_menu_id != menu_id:
                    if candidate_menu_id not in candidate_scores:
                        candidate_scores[candidate_menu_id] = 0
                    candidate_scores[candidate_menu_id] += (frequency / max_time_freq) * 0.3
        except:
            pass
        
        # 3. カテゴリ親和性スコア（重み: 20%）
        try:
            category_affinity = order_crud.get_category_cooccurrence(db, base_menu.category_id)
            max_category_freq = max([result[1] for result in category_affinity]) if category_affinity else 1
            
            # 親和性の高いカテゴリのメニューにスコアを追加
            for category_id, frequency in category_affinity:
                # 該当カテゴリのメニューを取得（基準メニューは除外）
                category_menus = db.query(Menu.id).filter(
                    Menu.category_id == category_id,
                    Menu.id != menu_id
                ).all()
                category_score = (frequency / max_category_freq) * 0.2
                
                for (menu_id_in_category,) in category_menus:
                    if menu_id_in_category not in candidate_scores:
                        candidate_scores[menu_id_in_category] = 0
                    candidate_scores[menu_id_in_category] += category_score
        except:
            pass
        
        if not candidate_scores:
            # フォールバック: Phase1を実行
            phase1 = Phase1FrequencyStrategy()
            return phase1.recommend(menu_id, db)
        
        # 最高スコアのメニューを返す
        best_menu_id = max(candidate_scores.items(), key=lambda x: x[1])[0]
        return best_menu_id