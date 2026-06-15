import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


np.random.seed(42)


def generate_simulated_traffic_data(days=30):
    """
    生成模拟交通流数据。

    参数：
        days: 模拟天数，默认生成30天数据

    返回：
        df: 交通流数据DataFrame
    """

    directions = ["East", "South", "West", "North"]
    start_time = datetime(2026, 6, 1, 0, 0, 0)

    records = []

    for day in range(days):
        for interval in range(288):  # 一天 24*60/5 = 288 个5分钟时间片
            current_time = start_time + timedelta(days=day, minutes=5 * interval)

            hour = current_time.hour
            minute = current_time.minute
            weekday = current_time.weekday()

            is_weekend = 1 if weekday >= 5 else 0

            is_morning_peak = 1 if 7 <= hour <= 9 else 0
            is_evening_peak = 1 if 17 <= hour <= 19 else 0
            is_peak = 1 if is_morning_peak or is_evening_peak else 0

            for direction in directions:
                if is_peak == 1:
                    vehicle_count = np.random.normal(90, 18)
                    avg_speed = np.random.normal(28, 8)
                    queue_length = np.random.normal(22, 8)
                    occupancy = np.random.normal(0.65, 0.12)
                else:
                    vehicle_count = np.random.normal(45, 15)
                    avg_speed = np.random.normal(48, 10)
                    queue_length = np.random.normal(8, 5)
                    occupancy = np.random.normal(0.35, 0.10)

                # 东西方向作为主干道，交通压力略大
                if direction in ["East", "West"]:
                    vehicle_count *= 1.10
                    queue_length *= 1.10
                    occupancy *= 1.05

                # 周末车流量略低，速度略高
                if is_weekend == 1:
                    vehicle_count *= 0.85
                    queue_length *= 0.85
                    occupancy *= 0.90
                    avg_speed *= 1.05

                # 限制数据范围，避免出现不合理值
                vehicle_count = max(5, min(150, vehicle_count))
                avg_speed = max(5, min(75, avg_speed))
                queue_length = max(0, min(60, queue_length))
                occupancy = max(0.05, min(0.95, occupancy))

                records.append({
                    "timestamp": current_time,
                    "date": current_time.date(),
                    "hour": hour,
                    "minute": minute,
                    "weekday": weekday,
                    "is_weekend": is_weekend,
                    "is_peak": is_peak,
                    "direction": direction,
                    "vehicle_count": round(vehicle_count, 2),
                    "avg_speed": round(avg_speed, 2),
                    "queue_length": round(queue_length, 2),
                    "occupancy": round(occupancy, 2)
                })

    df = pd.DataFrame(records)
    return df


def assign_congestion_label(row):
    """
    根据交通流特征构造拥堵状态标签。
    输出：畅通、缓行、拥堵
    """

    flow_score = row["vehicle_count"] / 150
    speed_score = (75 - row["avg_speed"]) / 75
    queue_score = row["queue_length"] / 60
    occupancy_score = row["occupancy"]

    congestion_score = (
        0.30 * flow_score +
        0.30 * speed_score +
        0.25 * queue_score +
        0.15 * occupancy_score
    )

    if congestion_score < 0.35:
        return "畅通"
    elif congestion_score < 0.60:
        return "缓行"
    else:
        return "拥堵"


def main():
    os.makedirs("data", exist_ok=True)

    df = generate_simulated_traffic_data(days=30)
    df["congestion_level"] = df.apply(assign_congestion_label, axis=1)

    save_path = "data/simulated_traffic_data.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("模拟交通数据生成完成！")
    print(f"保存路径：{save_path}")
    print("数据规模：", df.shape)
    print("\n前5行数据：")
    print(df.head())

    print("\n拥堵状态分布：")
    print(df["congestion_level"].value_counts())


if __name__ == "__main__":
    main()