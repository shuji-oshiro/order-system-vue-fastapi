#!/bin/bash

cd backend || exit 1

# 仮想環境がなければ作成
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

# 仮想環境をアクティブ化
source .venv/bin/activate || exit 1

# パッケージ同期
uv sync

cd ..

# Node.js 依存をインストール
npm install

# フロントエンド起動
cd frontend && npm install && npm run dev