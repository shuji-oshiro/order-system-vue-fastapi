-- レコメンドテスト用の注文データ
-- 座席IDのみを条件とする実装に最適化されたテストパターン

-- 座席1: 焼肉定食 + ビール の組み合わせ（高頻度パターン - 3回）
INSERT INTO orders (seat_id, menu_id, order_cnt, order_date) VALUES
(1, 11, 1, '2025-01-10 12:00:00'),  -- 焼肉定食
(1, 21, 1, '2025-01-10 12:05:00'),  -- ビール

-- 座席2: 同じ組み合わせを繰り返し（焼肉定食 + ビール）
(2, 11, 1, '2025-01-11 12:30:00'),  -- 焼肉定食
(2, 21, 2, '2025-01-11 12:35:00'),  -- ビール x2

-- 座席3: 同じ組み合わせを繰り返し（焼肉定食 + ビール）
(3, 11, 1, '2025-01-12 13:00:00'),  -- 焼肉定食
(3, 21, 1, '2025-01-12 13:05:00'),  -- ビール

-- 座席4: 唐揚げ + コーラ の組み合わせ（中頻度パターン - 2回）
(4, 1, 1, '2025-01-10 14:00:00'),   -- 唐揚げ
(4, 16, 1, '2025-01-10 14:05:00'),  -- コーラ

-- 座席5: 同じ組み合わせを繰り返し（唐揚げ + コーラ）
(5, 1, 1, '2025-01-11 14:30:00'),   -- 唐揚げ
(5, 16, 1, '2025-01-11 14:35:00'),  -- コーラ

-- 座席6: とんかつ定食 + アイスコーヒー の組み合わせ（低頻度パターン - 1回）
(6, 14, 1, '2025-01-10 13:00:00'),  -- とんかつ定食
(6, 19, 1, '2025-01-10 13:05:00'),  -- アイスコーヒー

-- 座席7: 複数メニューパターン（定食系の組み合わせテスト）
(7, 12, 1, '2025-01-10 12:00:00'),  -- 唐揚げ定食
(7, 17, 1, '2025-01-10 12:05:00'),  -- ウーロン茶
(7, 26, 1, '2025-01-10 12:20:00'),  -- プリン（デザート）

-- 座席8: おつまみ系の組み合わせ
(8, 2, 1, '2025-01-10 18:00:00'),   -- 枝豆
(8, 6, 1, '2025-01-10 18:05:00'),   -- 焼き餃子
(8, 22, 1, '2025-01-10 18:10:00'),  -- 日本酒

-- 座席9: 高価格帯メニューの組み合わせ（価格帯テスト用）
(9, 14, 1, '2025-01-10 19:00:00'),  -- とんかつ定食 (950円)
(9, 15, 1, '2025-01-10 19:05:00'),  -- ハンバーグ定食 (900円)
(9, 21, 1, '2025-01-10 19:10:00'),  -- ビール (400円)

-- 座席10: 低価格帯メニューの組み合わせ（価格帯テスト用）
(10, 2, 1, '2025-01-10 15:00:00'),  -- 枝豆 (300円)
(10, 5, 1, '2025-01-10 15:05:00'),  -- 冷奴 (250円)
(10, 16, 1, '2025-01-10 15:10:00'), -- コーラ (200円)

-- 座席11: 時間帯テスト用（朝食時間帯 - 8時台）
(11, 20, 1, '2025-01-10 08:00:00'), -- ホットコーヒー
(11, 26, 1, '2025-01-10 08:05:00'), -- プリン

-- 座席12: 時間帯テスト用（昼食時間帯 - 12時台）
(12, 11, 1, '2025-01-10 12:00:00'), -- 焼肉定食
(12, 17, 1, '2025-01-10 12:05:00'), -- ウーロン茶

-- 座席13: 時間帯テスト用（夜食時間帯 - 22時台）
(13, 8, 1, '2025-01-10 22:00:00'),  -- 漬物盛り合わせ
(13, 23, 1, '2025-01-10 22:05:00'), -- 焼酎

-- 座席14: 最近のトレンドテスト用（新しい注文 - 最近）
(14, 26, 1, '2025-01-15 20:00:00'), -- プリン
(14, 27, 1, '2025-01-15 20:05:00'), -- ケーキ

-- 座席15: 古い注文（トレンドテスト用 - 古い）
(15, 25, 1, '2025-01-01 20:00:00'), -- アイスクリーム
(15, 27, 1, '2025-01-01 20:05:00'), -- ケーキ

-- 座席16: カテゴリ親和性テスト用（定食カテゴリ + ドリンクカテゴリ）
(16, 13, 1, '2025-01-10 12:00:00'), -- 生姜焼き定食（定食カテゴリ）
(16, 18, 1, '2025-01-10 12:05:00'), -- オレンジジュース（ドリンクカテゴリ）

-- 座席17: カテゴリ親和性テスト用（おつまみカテゴリ + アルコールカテゴリ）
(17, 4, 1, '2025-01-10 18:00:00'),  -- だし巻き卵（おつまみカテゴリ）
(17, 24, 1, '2025-01-10 18:05:00'), -- ハイボール（アルコールカテゴリ）

-- 座席18: 複合テスト用（複数カテゴリの組み合わせ）
(18, 1, 1, '2025-01-10 19:00:00'),  -- 唐揚げ（おつまみ）
(18, 12, 1, '2025-01-10 19:05:00'), -- 唐揚げ定食（定食）
(18, 21, 1, '2025-01-10 19:10:00'), -- ビール（アルコール）
(18, 25, 1, '2025-01-10 19:20:00'); -- アイスクリーム（デザート）

-- 追加: より多くの頻度データを生成（焼肉定食 + ビールの組み合わせを強化）
INSERT INTO orders (seat_id, menu_id, order_cnt, order_date) VALUES
(19, 11, 1, '2025-01-13 12:00:00'), -- 焼肉定食
(19, 21, 1, '2025-01-13 12:05:00'), -- ビール

(20, 11, 1, '2025-01-14 12:30:00'), -- 焼肉定食
(20, 21, 1, '2025-01-14 12:35:00'), -- ビール

-- 唐揚げ + コーラの組み合わせも強化
(21, 1, 1, '2025-01-13 14:00:00'),  -- 唐揚げ
(21, 16, 1, '2025-01-13 14:05:00'); -- コーラ
