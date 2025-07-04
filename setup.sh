# 仮想環境をアクティブ化
backend/.venv/Scripts/activate &&

# パッケージ同期
cd backend && uv sync && cd ..

# Node.js 依存をインストール
npm install

# フロントエンド起動
cd frontend && npm install && npm run dev