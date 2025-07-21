"""
AI トレーニング API

モデルの学習指示を行うシンプルなAPI
実際のトレーニング処理は内部で実行される
"""
import logging
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from backend.app.database.database import get_db
from backend.app.service.ai_training.ai_model_trainer import AIModelTrainer


# Pydanticモデル定義
class TrainingRequest(BaseModel):
    """学習リクエストのスキーマ"""
    epochs: Optional[int] = 100
    batch_size: Optional[int] = 256
    learning_rate: Optional[float] = 0.001
    test_size: Optional[float] = 0.2
    patience: Optional[int] = 10
    force_reload: Optional[bool] = True


# APIルーター
router = APIRouter()


@router.get("/model/info")
async def get_model_info() -> Dict[str, Any]:
    """モデル情報を取得"""
    try:
        trainer = AIModelTrainer()
        info = trainer.get_model_info()

        if not info["metadata"]:
            raise HTTPException(status_code=404, detail="モデル情報が見つかりません")

        return {
            "status": "success",
            "data": info
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logging.error(f"モデル情報取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/exists")
async def check_model_exists() -> Dict[str, Any]:
    """学習済みモデルの存在チェック"""
    try:
        trainer = AIModelTrainer()
        exists = trainer.check_model_exists()
        return {
            "status": "success",
            "model_exists": exists,
            "message": "学習済みモデルが存在します" if exists else "学習済みモデルが見つかりません"
        }
    except Exception as e:
        logging.error(f"モデル存在チェックエラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_training_data(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """学習データの妥当性をチェック"""
    try:
        trainer = AIModelTrainer()
        validation_result = trainer.validate_training_data(db)
        return validation_result
    except Exception as e:
        logging.error(f"データ検証エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/new")
async def train_new_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> JSONResponse:
    """新規モデルを学習（バックグラウンド処理）"""
    try:
        trainer = AIModelTrainer()
        
        # データ検証
        validation_result = trainer.validate_training_data(db)
        if validation_result["status"] == "error":
            raise HTTPException(status_code=400, detail=validation_result["message"])
        
        if not validation_result["overall_ok"]:
            raise HTTPException(status_code=400, detail=validation_result["message"])
        
        # バックグラウンドで学習開始
        def train_model():
            success = trainer.train_new_model(
                db=db,
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                test_size=request.test_size,
                patience=request.patience,
                force_reload=request.force_reload
            )
            if success:
                logging.info("バックグラウンド学習が正常に完了しました")
            else:
                logging.error("バックグラウンド学習が失敗しました")
        
        background_tasks.add_task(train_model)
        
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "message": "新規モデルの学習をバックグラウンドで開始しました",
                "model_name": "neural_cf"
            }        
        )   
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"新規学習開始エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/retrain")
async def retrain_existing_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> JSONResponse:
    """既存モデルを再学習（バックグラウンド処理）"""
    try:
        trainer = AIModelTrainer()
        
        # 既存モデル存在チェック
        if not trainer.check_model_exists():
            raise HTTPException(
                status_code=404, 
                detail="既存モデルが見つかりません。新規学習を実行してください。"
            )
        
        modelinfo = trainer.get_model_info()  # モデル情報を取得してログに出力

        # バックグラウンドで再学習開始
        def retrain_model():
            success = trainer.retrain_existing_model(
                db=db,
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                test_size=request.test_size,
                patience=request.patience,
                force_reload=request.force_reload,
                modelinfo=modelinfo
            )
            if success:
                logging.info("バックグラウンド再学習が正常に完了しました")
            else:
                logging.error("バックグラウンド再学習が失敗しました")
        
        background_tasks.add_task(retrain_model)

        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "message": "既存モデルの再学習をバックグラウンドで開始しました",
                "model_name": "neural_cf"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"再学習開始エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/new/sync")
async def train_new_model_sync(
    request: TrainingRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """新規モデルを学習（同期処理）"""
    try:
        trainer = AIModelTrainer()
        
        # データ検証
        validation_result = trainer.validate_training_data(db)
        if validation_result["status"] == "error":
            raise HTTPException(status_code=400, detail=validation_result["message"])
        
        if not validation_result["overall_ok"]:
            raise HTTPException(status_code=400, detail=validation_result["message"])
        
        # 学習実行
        success = trainer.train_new_model(
            db=db,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            test_size=request.test_size,
            patience=request.patience,
            force_reload=request.force_reload
        )
        
        if success:
            return {
                "status": "success",
                "message": "新規モデルの学習が正常に完了しました",
                "model_name": "neural_cf"
            }
        else:
            return {
                "status": "error",
                "message": "モデルの学習に失敗しました",
                "model_name": "neural_cf"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"新規学習エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/model")
async def delete_model() -> Dict[str, Any]:
    """学習済みモデルを削除"""
    try:
        trainer = AIModelTrainer()
        
        if not trainer.check_model_exists():
            raise HTTPException(status_code=404, detail="削除対象のモデルが見つかりません")
        
        success = trainer.delete_model()
        
        if success:
            return {
                "status": "success",
                "message": "モデルの削除が正常に完了しました",
                "model_name": "neural_cf"
            }
        else:
            return {
                "status": "error",
                "message": "モデルの削除に失敗しました",
                "model_name": "neural_cf"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"モデル削除エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))
