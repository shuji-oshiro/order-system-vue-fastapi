import time
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi import APIRouter, Depends, HTTPException
from backend.app.database.database import get_db, engine
from backend.app.middleware.logging_middleware import metrics_middleware
from backend.app.utils.logging_config import get_logger

router = APIRouter()
logger = get_logger("monitoring")

# アプリケーション開始時刻
start_time = time.time()


@router.get("/health")
def health_check():
    """軽量なヘルスチェック（ログ・メトリクス対象外）"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime": round(time.time() - start_time, 2)
        }
    except Exception as e:
        # 軽量化のためログ出力は最小限に
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/health/simple")
def simple_health_check():
    """最軽量なヘルスチェック（監視システム用）"""
    return {"status": "ok"}


@router.get("/health/detailed")
def detailed_health_check(db: Session = Depends(get_db)):
    """詳細なヘルスチェック"""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime": round(time.time() - start_time, 2),
            "checks": {}
        }
        
        # データベース接続チェック
        try:
            db.execute(text("SELECT 1"))
            health_data["checks"]["database"] = {"status": "healthy"}
        except Exception as e:
            health_data["checks"]["database"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
            health_data["status"] = "degraded"
        
        # システムリソースチェック（基本版）
        try:
            import os
            import shutil
            
            # ディスク使用量チェック
            total, used, free = shutil.disk_usage('.')
            disk_percent = (used / total) * 100
            
            health_data["checks"]["system"] = {
                "status": "healthy",
                "disk_percent": round(disk_percent, 2)
            }
            
            # 閾値チェック
            if disk_percent > 80:
                health_data["checks"]["system"]["status"] = "degraded"
                health_data["status"] = "degraded"
                
        except Exception as e:
            health_data["checks"]["system"] = {
                "status": "unknown",
                "error": str(e)
            }
        
        return health_data
        
    except Exception as e:
        logger.error("詳細ヘルスチェックエラー", exc_info=True)
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.get("/metrics")
def get_metrics():
    """アプリケーションメトリクス"""
    try:
        # ミドルウェアからメトリクスを取得
        try:
            from backend.app.middleware.logging_middleware import MetricsMiddleware
            app_metrics = MetricsMiddleware.get_metrics()
        except Exception as metrics_error:
            logger.warning(f"メトリクス取得に失敗しました: {metrics_error}")
            app_metrics = {
                "request_count": 0,
                "error_count": 0,
                "error_rate": 0,
                "average_response_time": 0,
                "note": "メトリクス取得失敗"
            }
        
        # システムメトリクス（基本版）
        import os
        import shutil
        
        total, used, free = shutil.disk_usage('.')
        disk_percent = (used / total) * 100
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime": round(time.time() - start_time, 2),
            "application": app_metrics,
            "system": {
                "disk_total": total,
                "disk_free": free,
                "disk_used": used,
                "disk_percent": round(disk_percent, 2)
            }
        }
        
    except Exception as e:
        logger.error("メトリクス取得エラー", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/logs")
def get_recent_logs(lines: int = 100):
    """最新のログを取得（開発/デバッグ用）"""
    try:
        from pathlib import Path
        
        log_file = Path("./backend/logs/app.log")
        if not log_file.exists():
            return {"logs": [], "message": "ログファイルが見つかりません"}
        
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "logs": [line.strip() for line in recent_lines],
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines)
        }
        
    except Exception as e:
        logger.error("ログ取得エラー", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get logs")
