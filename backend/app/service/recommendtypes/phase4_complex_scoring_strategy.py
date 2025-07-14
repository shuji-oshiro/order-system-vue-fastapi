from sqlalchemy.orm import Session
from backend.app.crud import order_crud
from backend.app.models.model import Menu
from backend.app.service.recommendtypes.recommend_strategy import RecommendStrategy

class Phase4ComplexScoringStrategy(RecommendStrategy):
    """Phase 4: 複合スコアリングシステム"""
    
    def recommend(self, menu_id: int, db: Session) -> int:
        from datetime import datetime
        
        # 候補メニューのスコアを計算
        candidate_scores = {}
        
        # 1. 頻度スコア（重み: 40%）
        try:
            cooccurrence_results = order_crud.get_menu_cooccurrence_frequency(db, menu_id)
            max_freq = max([result[1] for result in cooccurrence_results]) if cooccurrence_results else 1
            
            for menu_id_candidate, frequency in cooccurrence_results:
                # 基準メニュー自身は除外
                if menu_id_candidate != menu_id:
                    if menu_id_candidate not in candidate_scores:
                        candidate_scores[menu_id_candidate] = 0
                    candidate_scores[menu_id_candidate] += (frequency / max_freq) * 0.4
        except:
            pass
        
        # 2. 時間帯スコア（重み: 20%）
        try:
            current_hour = datetime.now().hour
            popular_menus = order_crud.get_popular_menus_by_time_period(db, current_hour)
            max_time_freq = max([result[1] for result in popular_menus]) if popular_menus else 1
            
            for menu_id_candidate, frequency in popular_menus:
                # 基準メニュー自身は除外
                if menu_id_candidate != menu_id:
                    if menu_id_candidate not in candidate_scores:
                        candidate_scores[menu_id_candidate] = 0
                    candidate_scores[menu_id_candidate] += (frequency / max_time_freq) * 0.2
        except:
            pass
        
        # 3. 価格帯スコア（重み: 20%）
        try:
            base_menu = db.query(Menu).filter(Menu.id == menu_id).first()
            if base_menu and base_menu.price > 0:  # ゼロ除算回避
                price_range = int(base_menu.price * 0.3)
                min_price = max(0, base_menu.price - price_range)
                max_price = base_menu.price + price_range
                
                similar_price_menus = order_crud.get_menus_by_price_range(
                    db, min_price, max_price, exclude_menu_id=menu_id
                )
                
                for menu in similar_price_menus:
                    if menu.id not in candidate_scores:
                        candidate_scores[menu.id] = 0
                    # 価格差が小さいほど高スコア
                    price_diff_ratio = 1 - abs(menu.price - base_menu.price) / base_menu.price
                    candidate_scores[menu.id] += price_diff_ratio * 0.2
        except:
            pass
        
        # 4. 最近のトレンドスコア（重み: 20%）
        try:
            recent_popular = order_crud.get_recent_popular_menus(db, days=7)
            max_recent_freq = max([result[1] for result in recent_popular]) if recent_popular else 1
            
            for menu_id_candidate, frequency, _ in recent_popular:
                # 基準メニュー自身は除外
                if menu_id_candidate != menu_id:
                    if menu_id_candidate not in candidate_scores:
                        candidate_scores[menu_id_candidate] = 0
                    candidate_scores[menu_id_candidate] += (frequency / max_recent_freq) * 0.2
        except:
            pass
        
        if not candidate_scores:
            raise ValueError("推薦候補が見つかりません")
        
        # 最高スコアのメニューを返す
        best_menu_id = max(candidate_scores.items(), key=lambda x: x[1])[0]
        return best_menu_id