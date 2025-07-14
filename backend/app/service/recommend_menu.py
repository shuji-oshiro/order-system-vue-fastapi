from sqlalchemy.orm import Session
from backend.app.models.model import Menu
from backend.app.schemas.menu_schema import MenuOut
from backend.app.service.recommendtypes.phase1_frequency_strategy import Phase1FrequencyStrategy
from backend.app.service.recommendtypes.phase2_time_and_category_strategy import Phase2TimeAndCategoryStrategy
from backend.app.service.recommendtypes.phase3_price_consideration_strategy import Phase3PriceConsiderationStrategy
from backend.app.service.recommendtypes.phase4_complex_scoring_strategy import Phase4ComplexScoringStrategy
from backend.app.service.recommendtypes.phase5_ai_strategy import Phase5AIRecommendStrategy


def recommend_menu_by_menu_id(menu_id: int, db: Session, phase: int = 1) -> MenuOut:
    """
    メニューIDに基づいておすすめメニューを取得する
    
    Args:
        menu_id (int): 基準となるメニューID
        db (Session): データベースセッション
        phase (int): 実装フェーズ（1-5）
        
    Returns:
        MenuOut: おすすめメニュー
        
    Raises:
        ValueError: 該当する注文履歴やメニューが存在しない場合
    """
    # フェーズに応じた戦略を選択
    strategies = {
        1: Phase1FrequencyStrategy(),
        2: Phase2TimeAndCategoryStrategy(),
        3: Phase3PriceConsiderationStrategy(),
        4: Phase4ComplexScoringStrategy(),
        5: Phase5AIRecommendStrategy(),
    }
    
    if phase not in strategies:
        raise ValueError(f"無効なフェーズです: {phase}")
    
    strategy = strategies[phase]
    
    try:
        recommended_menu_id = strategy.recommend(menu_id, db)
        
        # メニュー情報を取得（カテゴリ情報も含む）
        menu = db.query(Menu).filter(Menu.id == recommended_menu_id).first()
        
        if not menu:
            raise ValueError(f"推薦メニューID {recommended_menu_id} が見つかりません")
        
        return MenuOut.model_validate(menu)
        
    except ValueError as e:
        # フォールバック: より基本的なフェーズにフォールバック
        if phase > 1:
            return recommend_menu_by_menu_id(menu_id, db, phase - 1)
        else:
            raise e


