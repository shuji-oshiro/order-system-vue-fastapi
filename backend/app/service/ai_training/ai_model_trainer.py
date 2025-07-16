"""
AI モデル トレーニング専用モジュール

Neural Collaborative Filtering モデルの学習・再学習・管理を行う
レコメンド処理とは完全に分離されている
"""
import os
import logging
import shutil
from typing import Dict, Any, Optional
from pathlib import Path
from sqlalchemy.orm import Session

from backend.app.crud import menu_crud
from backend.app.crud import order_crud
from backend.app.ml.models.neural_cf.model import NeuralCollaborativeFiltering
from backend.app.ml.inference.model_loader import ModelLoader
from backend.app.ml.utils.device_manager import DeviceManager


class AIModelTrainer:
    """AI モデルの学習・管理を行うクラス"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.device_manager = DeviceManager()
        self.model_name = "neural_cf"
        
    def check_model_exists(self) -> bool:
        """学習済みモデルが存在するかチェック"""
        try:
            available_models = self.model_loader.get_available_models(self.model_name)
            return "latest.pth" in available_models
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        available_models = self.model_loader.get_available_models(self.model_name)
        
        model_metadata = {}
        if "latest.pth" in available_models:
            try:
                model_metadata = self.model_loader.get_model_metadata(self.model_name)
            except Exception as e:
                logging.warning(f"メタデータ取得に失敗: {e}")
        
        return {
            "model_name": "Neural Collaborative Filtering",
            "model_storage_name": self.model_name,
            "model_exists": "latest.pth" in available_models,
            "available_models": available_models,
            "metadata": model_metadata,
            "device": str(self.device_manager.get_device())
        }
    
    def validate_training_data(self, db: Session) -> Dict[str, Any]:
        """
        学習データの妥当性をチェックする
        
        Args:
            db: データベースセッション
            
        Returns:
            バリデーション結果の辞書
        """
        try:
                        
            # メニュー数をチェック
            all_menus = menu_crud.get_all_menus(db)
            num_menus = len(all_menus)
            
            # 注文データをチェック
            all_orders = order_crud.get_all_orders(db)
            num_orders = len(all_orders)
            
            # 最低限のデータ要件チェック
            min_menus = 5
            min_orders = 10
            
            validation_results = {
                "num_menus": num_menus,
                "num_orders": num_orders,
                "min_menus_required": min_menus,
                "min_orders_required": min_orders,
                "menus_ok": num_menus >= min_menus,
                "orders_ok": num_orders >= min_orders,
                "overall_ok": num_menus >= min_menus and num_orders >= min_orders
            }
            
            if validation_results["overall_ok"]:
                message = "学習データは妥当です"
                status = "success"
            else:
                issues = []
                if not validation_results["menus_ok"]:
                    issues.append(f"メニュー数が不足（{num_menus} < {min_menus}）")
                if not validation_results["orders_ok"]:
                    issues.append(f"注文数が不足（{num_orders} < {min_orders}）")
                message = f"学習データに問題があります: {', '.join(issues)}"
                status = "warning"
            
            return {
                "status": status,
                "message": message,
                **validation_results
            }
            
        except Exception as e:
            logging.error(f"データ検証中にエラーが発生: {e}")
            return {
                "status": "error",
                "message": f"データ検証に失敗しました: {str(e)}"
            }
    
    def train_new_model(self, db: Session, **kwargs) -> bool:
        """
        新規モデルを学習する
        
        Args:
            db: データベースセッション
            **kwargs: 学習パラメータ
            
        Returns:
            bool: 学習成功フラグ
        """
        try:
            # 学習パラメータの設定
            epochs = kwargs.get('epochs', 100)
            batch_size = kwargs.get('batch_size', 256)
            learning_rate = kwargs.get('learning_rate', 0.001)
            test_size = kwargs.get('test_size', 0.2)
            patience = kwargs.get('patience', 10)
            force_reload = kwargs.get('force_reload', True)
            
            logging.info("Neural Collaborative Filteringモデルの学習を開始...")
            
            # モデル初期化（DBセッションを渡して完全に初期化）
            model = NeuralCollaborativeFiltering(db=db)
            
            # 学習実行
            train_results = model.fit(
                db=db,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                test_size=test_size,
                patience=patience,
                force_reload=force_reload
            )
            
            # モデル保存
            self.model_loader.save_model(model, self.model_name)
            logging.info(f"学習完了。モデルを保存: {self.model_name}")
            logging.info(f"最終学習損失: {train_results.get('final_train_loss', 0):.4f}")
            logging.info(f"最終検証損失: {train_results.get('final_val_loss', 0):.4f}")
            
            return True
            
        except Exception as e:
            logging.error(f"モデル学習中にエラーが発生: {e}")
            return False
    
    # def train_new_model(self, db: Session, **kwargs) -> bool:
    #     """
    #     新規モデルを学習する
        
    #     Args:
    #         db: データベースセッション
    #         **kwargs: 学習パラメータ
            
    #     Returns:
    #         bool: 学習成功フラグ
    #     """
    #     return self._execute_training(db, **kwargs)
    
    def retrain_existing_model(self, db: Session, **kwargs) -> bool:
        """
        既存モデルを再学習する（追加学習）
        
        Args:
            db: データベースセッション
            **kwargs: 学習パラメータ
            
        Returns:
            bool: 学習成功フラグ
        """
        if not self.check_model_exists():
            logging.error("既存モデルが見つかりません")
            return False
        
        logging.info("既存モデルの再学習を開始...")
        
        try:
            # 既存モデルを読み込み
            device = self.device_manager.get_device()
            model = NeuralCollaborativeFiltering(db=db)
            
            # モデル読み込み
            loaded_model = self.model_loader.load_model(model, self.model_name, device=device)
            if not isinstance(loaded_model, NeuralCollaborativeFiltering):
                raise ValueError("モデルの読み込みに失敗しました")
            
            # 追加学習実行
            train_results = loaded_model.fit(
                db=db,
                # force_reload=kwargs.get('force_reload', True),
                **kwargs
            )
            
            # モデル保存
            self.model_loader.save_model(loaded_model, self.model_name)
            logging.info(f"再学習完了。モデルを更新保存: {self.model_name}")
            
            return True
            
        except Exception as e:
            logging.error(f"モデル再学習中にエラーが発生: {e}")
            return False
    
    def delete_model(self) -> bool:
        """
        学習済みモデルを削除する
        
        Returns:
            bool: 削除成功フラグ
        """
        try:
            # saved_models ディレクトリから削除
            model_dir = Path(self.model_loader.model_dir) / self.model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logging.info(f"モデルディレクトリを削除: {model_dir}")
            
            # GPU メモリクリア
            self.device_manager.clear_cache()
            
            return True
            
        except Exception as e:
            logging.error(f"モデル削除中にエラーが発生: {e}")
            return False
