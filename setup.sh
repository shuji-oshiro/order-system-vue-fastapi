#!/bin/bash

cd backend || exit 1

# 仮想環境で必要パッケージ同期
uv sync || exit 1

cd ..

# Node.js 依存をインストール
npm install || exit 1

# フロントエンド起動
cd frontend && npm install