#!/bin/bash

cd order-system-vue-fastapi || exit 1

# 仮想環境をアクティブ化
.venv\Scripts\Activate.ps1

# パッケージ同期
cd backend && uv sync && cd ..

# Node.js 依存をインストール
npm install

# フロントエンド起動
cd frontend && npm install && npm run dev