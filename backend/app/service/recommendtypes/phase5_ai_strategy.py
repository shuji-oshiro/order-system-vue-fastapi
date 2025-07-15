"""
Phase5: AI学習アルゴリズム戦略クラス

Neural Collaborative Filteringを使用した高度なレコメンド戦略
"""
import os
import logging
from typing import Dict, Any
from sqlalchemy.orm import Session
from backend.app.models.model import Menu
from backend.app.ml.models.neural_cf.model import NeuralCollaborativeFiltering
from backend.app.service.recommendtypes.recommend_strategy import RecommendStrategy

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
        
        # データベースから全メニューの数を取得
        all_menus = db.query(Menu).all()
        num_menus = len(all_menus)
        
        if not all_menus:
            raise ValueError("メニューデータが見つかりません")
        
        # モデル初期化（キャッシュ機能付き）
        self.model = NeuralCollaborativeFiltering(
            num_menus=num_menus,
            embedding_dim=64,
            hidden_dims=[128, 64, 32],
            dropout_rate=0.3,
            db=db  # 初期化時にデータをキャッシュ
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
        
        # 新規学習（キャッシュ使用で高速化）
        logging.info("Neural Collaborative Filteringモデルの学習を開始...")
        
        try:
            # キャッシュの状態をチェックして、dbを渡すかどうかを決定
            needs_db = (not hasattr(self.model, '_cached_orders') or 
                       self.model._cached_orders is None or 
                       not self.model._is_cache_valid())
            
            if needs_db:
                # キャッシュが無効な場合はdbを渡す
                logging.info("キャッシュが無効なため、DBセッションを渡して学習を実行")
                train_results = self.model.fit(
                    db=db,
                    epochs=50,
                    batch_size=128,
                    learning_rate=0.001,
                    test_size=0.2,
                    force_reload=False
                )
            else:
                # キャッシュが有効な場合はdbを渡さない
                logging.info("キャッシュが有効なため、キャッシュデータで学習を実行")
                train_results = self.model.fit(
                    epochs=50,
                    batch_size=128,
                    learning_rate=0.001,
                    test_size=0.2,
                    force_reload=False
                )
        except Exception as e:
            logging.error(f"モデル学習中にエラーが発生: {e}")
            raise ValueError("モデルの学習に失敗しました。データを確認してください。")
            
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
            
            # メニュー間関連性に基づく推薦の実行
            recommendations = model.recommend_similar_menus(
                base_menu_id=menu_id,
                exclude_menu_ids=[menu_id],
                top_k=5
            )
            
            if not recommendations:
                raise ValueError("AI推薦に失敗しました")
            
            # 最高スコアのメニューを返す
            recommended_menu_id, score = recommendations[0]
            
            logging.info(f"AI推薦完了: メニューID {recommended_menu_id} (関連度スコア: {score:.4f})")
            
            return recommended_menu_id
            
        except Exception as e:
            logging.error(f"AI推薦でエラーが発生: {e}")
            # フォールバック: Phase4にフォールバック
            from ..recommend_menu import Phase4ComplexScoringStrategy
            fallback_strategy = Phase4ComplexScoringStrategy()
            return fallback_strategy.recommend(menu_id, db)
    
    def retrain_model(self, db: Session, **kwargs) -> Dict[str, Any]:
        """
        モデルを再学習する
        
        Args:
            db: データベースセッション
            **kwargs: 学習パラメータ
                - force_reload: 強制的にデータを再読み込み（デフォルト: True）
            
        Returns:
            学習結果
        """
        # 既存モデルを削除
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        
        # キャッシュもクリア
        if self.model:
            self.model.clear_cache()
        
        self.model = None
        
        # 強制データ再読み込みで新規学習
        force_reload = kwargs.get('force_reload', True)
        
        # 新規学習
        model = self._load_or_train_model(db)
        
        # 強制再読み込みが指定されている場合は追加学習
        if force_reload and model:
            logging.info("強制データ再読み込みで追加学習を実行...")
            additional_results = model.fit(
                db=db,
                force_reload=True,
                **kwargs
            )
            
        return {
            "status": "success",
            "message": "モデルの再学習が完了しました",
            "model_path": self.model_path,
            "cache_cleared": True
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
