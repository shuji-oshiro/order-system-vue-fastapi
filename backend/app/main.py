from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.models import model
from backend.app.database import database
from backend.app.api import menu_api, order_api, category_api, menulist_api, voice_api
from backend.app.api.auth import routes as auth_routes
# データベースのテーブルを作成
# これにより、models.pyで定義したテーブルがデータベースに作成されます。
# もしテーブルが既に存在する場合は何も行いません。
model.Base.metadata.create_all(bind=database.engine)

app = FastAPI()


# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーター登録
# ルーターを登録して、各APIエンドポイントを設定します。
app.include_router(menu_api.router, prefix="/menu", tags=["menu"])
app.include_router(menulist_api.router, prefix="/menulist", tags=["menulist"])
app.include_router(order_api.router, prefix="/order", tags=["order"])
app.include_router(voice_api.router, prefix="/voice", tags=["voice"])
app.include_router(category_api.router, prefix="/category", tags=["category"])
app.include_router(auth_routes.router, prefix="/auth", tags=["auth"])


