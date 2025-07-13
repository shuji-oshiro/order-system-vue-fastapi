"""
Phase5: AI学習アルゴリズム戦略クラス

Neural Collaborative Filteringを使用した高度なレコメンド戦略
"""
import os
from typing import Dict, Any
from sqlalchemy.orm import Session
from ..service.recommend_menu import RecommendStrategy
from ..ml.models.neural_cf import NeuralCollaborativeFiltering
from ..models.model import Menu
import logging


class Phase5AIRecommendStrategy(RecommendStrategy):
    """Phase 5: AI学習アルゴリズム（Neural Collaborative Filtering）"""
    
    def __init__(self):
        self.model = None
        self.model_path = "backend/app/ml/saved_models/neural_cf_model.pth"
        self._ensure_model_directory()
        
    def _ensure_model_directory(self):
        """モデル保存ディレクトリを作成"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def _load_or_train_model(self, db: Session) -> NeuralCollaborativeFiltering:
        """
        学習済みモデルを読み込むか、新規に学習する
        """
        if self.model is not None and self.model.is_trained:
            return self.model
        
        # データベースから全メニューと座席の数を取得
        all_menus = db.query(Menu).all()
        
        # 座席IDの最大値を取得（動的に設定）
        from backend.app.crud import order_crud
        all_orders = order_crud.get_all_orders(db)
        
        if not all_orders:
            raise ValueError("注文データが見つかりません")
        
        max_seat_id = max([order.seat_id for order in all_orders])
        num_users = max_seat_id + 1  # 0-indexedを考慮
        num_items = len(all_menus)
        
        # モデル初期化
        self.model = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=64,
            hidden_dims=[128, 64, 32],
            dropout_rate=0.3
        )
        
        # 既存の学習済みモデルがあるか確認
        if os.path.exists(self.model_path):
            try:
                logging.info(f"学習済みモデルを読み込み: {self.model_path}")
                self.model.load_model(self.model_path)
                return self.model
            except Exception as e:
                logging.warning(f"モデル読み込みに失敗: {e}")
                logging.info("新規学習を開始します")
        
        # 新規学習
        logging.info("Neural Collaborative Filteringモデルの学習を開始...")
        
        train_results = self.model.train(
            db=db,
            epochs=50,
            batch_size=128,
            learning_rate=0.001,
            test_size=0.2
        )
        
        # モデル保存
        self.model.save_model(self.model_path)
        logging.info(f"学習完了。モデルを保存: {self.model_path}")
        logging.info(f"最終学習損失: {train_results['final_train_loss']:.4f}")
        logging.info(f"最終検証損失: {train_results['final_test_loss']:.4f}")
        
        return self.model
    
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
            # モデルを読み込みまたは学習
            model = self._load_or_train_model(db)
            
            # 基準メニューを注文した座席を特定
            from backend.app.crud import order_crud
            
            # 最新の座席IDを取得（実際のアプリケーションでは現在の座席IDを使用）
            from sqlalchemy import func
            from backend.app.models.model import Order
            
            recent_order = db.query(
                func.max(Order.seat_id)
            ).scalar()
            
            if not recent_order:
                # フォールバック: 全座席の中から最も頻繁に基準メニューを注文している座席を使用
                seat_orders = order_crud.get_seats_by_menu_id(db, menu_id)
                if not seat_orders:
                    raise ValueError("基準メニューの注文履歴が見つかりません")
                
                # 最も多く注文している座席を使用
                seat_counts = {}
                for (seat_id,) in seat_orders:
                    seat_counts[seat_id] = seat_counts.get(seat_id, 0) + 1
                
                user_id = max(seat_counts.items(), key=lambda x: x[1])[0]
            else:
                user_id = recent_order
            
            # AI推薦の実行
            recommendations = model.recommend(
                user_id=user_id,
                exclude_menu_ids=[menu_id],
                top_k=5
            )
            
            if not recommendations:
                raise ValueError("AI推薦に失敗しました")
            
            # 最高スコアのメニューを返す
            recommended_menu_id, score = recommendations[0]
            
            logging.info(f"AI推薦完了: メニューID {recommended_menu_id} (スコア: {score:.4f})")
            
            return recommended_menu_id
            
        except Exception as e:
            logging.error(f"AI推薦でエラーが発生: {e}")
            # フォールバック: Phase4にフォールバック
            from ..service.recommend_menu import Phase4ComplexScoringStrategy
            fallback_strategy = Phase4ComplexScoringStrategy()
            return fallback_strategy.recommend(menu_id, db)
    
    def retrain_model(self, db: Session, **kwargs) -> Dict[str, Any]:
        """
        モデルを再学習する
        
        Args:
            db: データベースセッション
            **kwargs: 学習パラメータ
            
        Returns:
            学習結果
        """
        # 既存モデルを削除
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        
        self.model = None
        
        # 新規学習
        model = self._load_or_train_model(db)
        
        return {
            "status": "success",
            "message": "モデルの再学習が完了しました",
            "model_path": self.model_path
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        モデル情報を取得
        """
        return {
            "model_name": "Neural Collaborative Filtering",
            "model_path": self.model_path,
            "is_trained": self.model.is_trained if self.model else False,
            "model_exists": os.path.exists(self.model_path)
        }
