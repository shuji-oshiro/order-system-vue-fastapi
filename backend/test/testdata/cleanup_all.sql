-- データベース全体クリーンアップ用SQL
-- 外部キー制約を考慮した順序で全テーブルをクリーンアップ

-- 注文データを削除（最も依存される側）
DELETE FROM orders;

-- メニューデータを削除（カテゴリに依存）
DELETE FROM menus;

-- カテゴリデータを削除（最も依存する側）
DELETE FROM categories;
