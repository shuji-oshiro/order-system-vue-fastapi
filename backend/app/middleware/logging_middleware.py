import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from backend.app.utils.logging_config import get_logger

logger = get_logger("middleware")


# アプリケーションの性能や使用状況を定量的に可視化
class LoggingMiddleware(BaseHTTPMiddleware):
    """リクエスト/レスポンスログ記録ミドルウェア"""
    
    # ログ出力を除外するパス（ヘルスチェック等の監視用エンドポイント）
    EXCLUDED_PATHS = {
        "/monitoring/health",
        "/monitoring/health/simple",
        "/monitoring/health/detailed",
        "/monitoring/metrics",
        "/monitoring/logs",
        "/docs",
        "/redoc", 
        "/openapi.json"
    }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # リクエストIDを生成
        request_id = str(uuid.uuid4())
        
        # リクエスト開始時刻
        start_time = time.time()
        
        # 除外パスの場合はログ出力をスキップ
        should_log = request.url.path not in self.EXCLUDED_PATHS
        
        # リクエスト情報をログ（除外パス以外）
        if should_log:
            logger.info(
                f"リクエスト開始: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "endpoint": str(request.url.path),
                    "query_params": str(request.query_params),
                    "client_ip": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", "unknown")
                }
            )
        
        # レスポンス処理
        try:
            response = await call_next(request)
            response_time = time.time() - start_time
            
            # レスポンス情報をログ（除外パス以外）
            if should_log:
                logger.info(
                    f"リクエスト完了: {request.method} {request.url.path} - {response.status_code}",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "endpoint": str(request.url.path),
                        "status_code": response.status_code,
                        "response_time": round(response_time * 1000, 2)  # ミリ秒
                    }
                )
            
            # レスポンスヘッダーにリクエストIDを追加
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # エラーログ（除外パス以外）
            if should_log:
                logger.error(
                    f"リクエストエラー: {request.method} {request.url.path} - {str(e)}",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "endpoint": str(request.url.path),
                        "response_time": round(response_time * 1000, 2),
                        "error": str(e)
                    },
                    exc_info=True
                )
            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """メトリクス収集ミドルウェア"""
    
    # 監視系エンドポイントを除外するパス
    EXCLUDED_PATHS = {
        "/monitoring/health",
        "/monitoring/health/simple",
        "/monitoring/health/detailed", 
        "/monitoring/metrics",
        "/monitoring/logs",
        "/docs",
        "/redoc",
        "/openapi.json"
    }
    
    # クラス変数でメトリクスを管理（全インスタンス共通）
    _request_count = 0          # 全リクエスト数
    _business_request_count = 0  # ビジネスロジック関連のリクエスト数
    _error_count = 0
    _business_error_count = 0    # ビジネスロジック関連のエラー数
    _total_response_time = 0.0
    _business_response_time = 0.0  # ビジネスロジック関連のレスポンス時間
    
    def __init__(self, app):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        is_business_request = request.url.path not in self.EXCLUDED_PATHS
        
        try:
            response = await call_next(request)
            
            # メトリクス更新（クラス変数を使用）
            MetricsMiddleware._request_count += 1
            response_time = time.time() - start_time
            MetricsMiddleware._total_response_time += response_time
            
            # ビジネスロジック関連のメトリクス更新
            if is_business_request:
                MetricsMiddleware._business_request_count += 1
                MetricsMiddleware._business_response_time += response_time
            
            if response.status_code >= 400:
                MetricsMiddleware._error_count += 1
                if is_business_request:
                    MetricsMiddleware._business_error_count += 1
                
            return response
            
        except Exception as e:
            MetricsMiddleware._request_count += 1
            MetricsMiddleware._error_count += 1
            response_time = time.time() - start_time
            MetricsMiddleware._total_response_time += response_time
            
            # ビジネスロジック関連のメトリクス更新
            if is_business_request:
                MetricsMiddleware._business_request_count += 1
                MetricsMiddleware._business_error_count += 1
                MetricsMiddleware._business_response_time += response_time
            raise
    
    @classmethod
    def get_metrics(cls):
        """現在のメトリクスを取得"""
        # 全体のメトリクス
        avg_response_time = (
            cls._total_response_time / cls._request_count 
            if cls._request_count > 0 
            else 0
        )
        
        # ビジネスロジック関連のメトリクス
        business_avg_response_time = (
            cls._business_response_time / cls._business_request_count
            if cls._business_request_count > 0
            else 0
        )
        
        return {
            "total": {
                "request_count": cls._request_count,
                "error_count": cls._error_count,
                "error_rate": cls._error_count / cls._request_count if cls._request_count > 0 else 0,
                "average_response_time": round(avg_response_time * 1000, 2)  # ミリ秒
            },
            "business": {
                "request_count": cls._business_request_count,
                "error_count": cls._business_error_count,
                "error_rate": cls._business_error_count / cls._business_request_count if cls._business_request_count > 0 else 0,
                "average_response_time": round(business_avg_response_time * 1000, 2)  # ミリ秒
            }
        }

# グローバルメトリクスアクセス用（後方互換性のため）
metrics_middleware = type('MetricsProxy', (), {
    'get_metrics': lambda: MetricsMiddleware.get_metrics()
})()
