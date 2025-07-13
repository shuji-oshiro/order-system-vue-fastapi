import random
from typing import List, Tuple
from sqlalchemy.orm import Session
from backend.app.models.model import Menu
from backend.app.schemas.menu_schema import MenuOut
from backend.app.crud import order_crud


class RecommendStrategy:
    """レコメンド戦略の基底クラス"""
    
    def recommend(self, menu_id: int, db: Session) -> int:
        """
        推薦メニューIDを返す
        
        Returns:
            int: 推薦するメニューID
        """
        raise NotImplementedError


class Phase1FrequencyStrategy(RecommendStrategy):
    """Phase 1: 頻度ベースレコメンド"""
    
    def recommend(self, menu_id: int, db: Session) -> int:
        # 指定メニューと一緒に注文されたメニューの頻度を取得
        cooccurrence_results = order_crud.get_menu_cooccurrence_frequency(db, menu_id)
        
        if not cooccurrence_results:
            raise ValueError("共起するメニューが見つかりません")
        
        # 最も頻度の高いメニューを返す
        return cooccurrence_results[0][0]  # (menu_id, frequency) の最初の要素


class Phase2TimeAndCategoryStrategy(RecommendStrategy):
    """Phase 2: 時間帯 + カテゴリ親和性レコメンド"""
    
    def recommend(self, menu_id: int, db: Session) -> int:
        from datetime import datetime
        
        # まず頻度ベースを試行
        try:
            phase1 = Phase1FrequencyStrategy()
            return phase1.recommend(menu_id, db)
        except ValueError:
            pass
        
        # 現在時刻での人気メニューを取得
        current_hour = datetime.now().hour
        popular_menus = order_crud.get_popular_menus_by_time_period(db, current_hour)
        
        if popular_menus:
            return popular_menus[0][0]
        
        # カテゴリ親和性も試行（実装は後で詳細化）
        raise ValueError("推薦できるメニューが見つかりません")


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
            # 頻度ベースで絞り込み
            try:
                phase1 = Phase1FrequencyStrategy()
                freq_recommendation = phase1.recommend(menu_id, db)
                
                # 頻度ベースの推薦が価格帯に含まれているかチェック
                for menu in similar_price_menus:
                    if menu.id == freq_recommendation:
                        return freq_recommendation
            except ValueError:
                pass
            
            # 価格帯メニューからランダム選択
            return random.choice(similar_price_menus).id
        
        # フォールバック: Phase2を実行
        phase2 = Phase2TimeAndCategoryStrategy()
        return phase2.recommend(menu_id, db)


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
                if menu_id_candidate not in candidate_scores:
                    candidate_scores[menu_id_candidate] = 0
                candidate_scores[menu_id_candidate] += (frequency / max_time_freq) * 0.2
        except:
            pass
        
        # 3. 価格帯スコア（重み: 20%）
        try:
            base_menu = db.query(Menu).filter(Menu.id == menu_id).first()
            if base_menu:
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


def recommend_menu_by_menu_id(menu_id: int, db: Session, phase: int = 1) -> MenuOut:
    """
    メニューIDに基づいておすすめメニューを取得する
    
    Args:
        menu_id (int): 基準となるメニューID
        db (Session): データベースセッション
        phase (int): 実装フェーズ（1-4）
        
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


