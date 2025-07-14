from sqlalchemy.orm import Session

class RecommendStrategy:
    """レコメンド戦略の基底クラス"""
    
    def recommend(self, menu_id: int, db: Session) -> int:
        """
        推薦メニューIDを返す
        
        Returns:
            int: 推薦するメニューID
        """
        raise NotImplementedError