#!/usr/bin/env python3
"""
注文ダミーデータ生成スクリプト

このスクリプトは注文テーブル用のダミーデータを生成します。
実行方法:
    python scripts/generate_order_dummy_data.py
"""

import random
from datetime import datetime, timedelta
from pathlib import Path


def generate_order_dummy_data(num_orders=10, output_file=None):
    """
    注文ダミーデータを生成してSQLファイルに出力
    
    Args:
        num_orders (int): 生成する注文数
        output_file (str): 出力するSQLファイルのパス
    """
    
    # メニューIDの範囲（insert_menus.sqlに基づく）
    menu_ids = {
        'おつまみ': list(range(1, 11)),     # カテゴリ1: 1-10
        '定食': list(range(11, 16)),        # カテゴリ2: 11-15
        'ドリンク': list(range(16, 21)),    # カテゴリ3: 16-20
        'アルコール': list(range(21, 25)),  # カテゴリ4: 21-24
        'デザート': list(range(25, 28))     # カテゴリ5: 25-27
    }
    
    # 全メニューID
    all_menu_ids = []
    for ids in menu_ids.values():
        all_menu_ids.extend(ids)
    
    # 座席数の範囲
    seat_range = range(1, 9)  # 1-8番席
    
    # 注文数の重み付け（現実的な分布）
    order_count_weights = {
        1: 0.5,   # 1個: 50%
        2: 0.3,   # 2個: 30%
        3: 0.15,  # 3個: 15%
        4: 0.05   # 4個: 5%
    }
    
    # 時間帯別の重み付け
    time_periods = {
        'lunch': (11, 14),      # ランチタイム 11:00-14:00
        'afternoon': (14, 17),  # 午後 14:00-17:00
        'dinner': (17, 22)      # ディナータイム 17:00-22:00
    }
    
    # 基準日（今日から過去7日間）
    base_date = datetime.now()
    date_range = [base_date - timedelta(days=i) for i in range(7)]
    
    orders = []
    
    for i in range(num_orders):
        # ランダムな座席を選択
        seat_id = random.choice(seat_range)
        
        # ランダムなメニューを選択
        menu_id = random.choice(all_menu_ids)
        
        # 注文数を重み付きランダムで選択
        order_counts = list(order_count_weights.keys())
        weights = list(order_count_weights.values())
        order_cnt = random.choices(order_counts, weights=weights)[0]
        
        # ランダムな日付と時間を生成
        order_date = random.choice(date_range)
        
        # 時間帯をランダムに選択
        period = random.choice(list(time_periods.keys()))
        hour_range = time_periods[period]
        hour = random.randint(hour_range[0], hour_range[1] - 1)
        minute = random.randint(0, 59)
        
        order_datetime = order_date.replace(
            hour=hour, 
            minute=minute, 
            second=0, 
            microsecond=0
        )
        
        orders.append({
            'seat_id': seat_id,
            'menu_id': menu_id,
            'order_cnt': order_cnt,
            'order_date': order_datetime.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # SQLファイルを生成
    sql_content = generate_sql_content(orders)
    
    # 出力ファイルのパスを決定
    if output_file is None:
        script_dir = Path(__file__).parent
        output_file = script_dir.parent / "test" / "testdata" / "insert_orders_generated.sql"
    
    # ファイルに書き込み
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(sql_content)
    
    print(f"注文ダミーデータを生成しました: {output_file}")
    print(f"生成された注文数: {len(orders)}")
    
    return orders, output_file


def generate_sql_content(orders):
    """
    注文データからSQL文を生成
    
    Args:
        orders (list): 注文データのリスト
        
    Returns:
        str: SQL文
    """
    
    sql_lines = [
        "-- 注文ダミーデータ（自動生成）",
        "-- 既存の注文データを削除してから新しいデータを挿入",
        "DELETE FROM orders;",
        "",
        "INSERT INTO orders (seat_id, menu_id, order_cnt, order_date) VALUES"
    ]
    
    # 注文データをSQL形式に変換
    value_lines = []
    for order in orders:
        value_line = f"({order['seat_id']}, {order['menu_id']}, {order['order_cnt']}, '{order['order_date']}')"
        value_lines.append(value_line)
    
    # 最後の行以外はカンマを付ける
    for i, line in enumerate(value_lines):
        if i < len(value_lines) - 1:
            sql_lines.append(line + ",")
        else:
            sql_lines.append(line + ";")
    
    return "\n".join(sql_lines)


def generate_realistic_scenario_data():
    """
    より現実的なシナリオベースの注文データを生成
    """
    scenarios = [
        # シナリオ1: ランチタイムの定食注文
        {
            'seat_id': 1,
            'orders': [
                {'menu_id': 11, 'order_cnt': 1},  # 焼肉定食
                {'menu_id': 19, 'order_cnt': 1}   # アイスコーヒー
            ],
            'time': '12:30:00'
        },
        # シナリオ2: 夜の飲み会
        {
            'seat_id': 2,
            'orders': [
                {'menu_id': 1, 'order_cnt': 2},   # 唐揚げ x2
                {'menu_id': 2, 'order_cnt': 1},   # 枝豆
                {'menu_id': 21, 'order_cnt': 4}   # ビール x4
            ],
            'time': '19:15:00'
        },
        # シナリオ3: カップルのディナー
        {
            'seat_id': 3,
            'orders': [
                {'menu_id': 15, 'order_cnt': 1},  # ハンバーグ定食
                {'menu_id': 12, 'order_cnt': 1},  # 唐揚げ定食
                {'menu_id': 18, 'order_cnt': 2},  # オレンジジュース x2
                {'menu_id': 26, 'order_cnt': 2}   # プリン x2
            ],
            'time': '19:30:00'
        }
    ]
    
    orders = []
    base_date = datetime.now().strftime('%Y-%m-%d')
    
    for scenario in scenarios:
        for order in scenario['orders']:
            orders.append({
                'seat_id': scenario['seat_id'],
                'menu_id': order['menu_id'],
                'order_cnt': order['order_cnt'],
                'order_date': f"{base_date} {scenario['time']}"
            })
    
    return orders


if __name__ == "__main__":
    # ランダムデータ生成
    print("=== ランダム注文データ生成 ===")
    generate_order_dummy_data(num_orders=15)
    
    print("\n=== シナリオベース注文データ生成 ===")
    # シナリオベースデータ生成
    scenario_orders = generate_realistic_scenario_data()
    sql_content = generate_sql_content(scenario_orders)
    
    script_dir = Path(__file__).parent
    scenario_file = script_dir.parent / "test" / "testdata" / "insert_orders_scenario.sql"
    
    with open(scenario_file, 'w', encoding='utf-8') as f:
        f.write(sql_content)
    
    print(f"シナリオベース注文データを生成しました: {scenario_file}")
    print(f"生成された注文数: {len(scenario_orders)}")
