from sqlalchemy.orm import Session
from backend.app.crud import order_crud
from backend.app.models.model import Menu
from backend.app.service.recommendtypes.recommend_strategy import RecommendStrategy
from backend.app.service.recommendtypes.phase1_frequency_strategy import Phase1FrequencyStrategy
from backend.app.service.recommendtypes.phase2_time_and_category_strategy import Phase2TimeAndCategoryStrategy

class Phase3PriceConsiderationStrategy(RecommendStrategy):
    """Phase 3: 価格帯考慮レコメンド"""
    
    def recommend(self, menu_id: int, db: Session) -> int:
        # 基準メニューの価格を取得
        base_menu = db.query(Menu).filter(Menu.id == menu_id).first()
        if not base_menu:
            raise ValueError("基準メニューが見つかりません")
        
        # 価格帯の範囲を設定（±30%）
        price_range = int(base_menu.price * 0.3)
        min_price = max(0, base_menu.price - price_range)
        max_price = base_menu.price + price_range
        
        # 同価格帯のメニューを取得
        similar_price_menus = order_crud.get_menus_by_price_range(
            db, min_price, max_price, exclude_menu_id=menu_id
        )
        
        if similar_price_menus:
            # Phase2のスコアリングシステムを使用して価格帯内で最適な選択
            phase2_strategy = Phase2TimeAndCategoryStrategy()
            try:
                phase2_recommendation = phase2_strategy.recommend(menu_id, db)
                
                # Phase2の推薦が価格帯に含まれているかチェック
                for menu in similar_price_menus:
                    if menu.id == phase2_recommendation:
                        return phase2_recommendation
            except ValueError:
                pass
            
            # Phase2で推薦されなかった場合、頻度ベースを試行
            try:
                phase1 = Phase1FrequencyStrategy()
                freq_recommendation = phase1.recommend(menu_id, db)
                
                # 頻度ベースの推薦が価格帯に含まれているかチェック
                for menu in similar_price_menus:
                    if menu.id == freq_recommendation:
                        return freq_recommendation
            except ValueError:
                pass
            
            # 最後の手段として、価格帯内で最も共起頻度の高いメニューを選択
            candidate_scores = {}
            try:
                cooccurrence_results = order_crud.get_menu_cooccurrence_frequency(db, menu_id)
                for candidate_menu_id, frequency in cooccurrence_results:
                    # 価格帯内のメニューのみスコア付与
                    for menu in similar_price_menus:
                        if menu.id == candidate_menu_id:
                            candidate_scores[candidate_menu_id] = frequency
                            break
                
                if candidate_scores:
                    return max(candidate_scores.items(), key=lambda x: x[1])[0]
            except:
                pass
            
            # 全て失敗した場合、価格差が最小のメニューを選択
            closest_price_menu = min(similar_price_menus, 
                                   key=lambda m: abs(m.price - base_menu.price))
            return closest_price_menu.id
        
        # フォールバック: Phase2を実行
        phase2 = Phase2TimeAndCategoryStrategy()
        return phase2.recommend(menu_id, db)