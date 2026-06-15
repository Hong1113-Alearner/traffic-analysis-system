import os
import joblib

import pandas as pd


# ==============================
# 1. 路径设置
# ==============================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "simulated_traffic_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "models", "label_mapping.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# 2. 模型需要的特征列
# ==============================

FEATURE_COLUMNS = [
    "vehicle_count",
    "avg_speed",
    "queue_length",
    "occupancy",
    "hour",
    "weekday",
    "is_weekend",
    "is_peak",
    "direction"
]


# ==============================
# 3. 加载数据和模型
# ==============================

def load_data_and_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"未找到数据文件：{DATA_PATH}\n"
            f"请先运行 src/01_generate_data.py"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"未找到模型文件：{MODEL_PATH}\n"
            f"请先运行 src/03_train_model.py"
        )

    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(
            f"未找到标签映射文件：{LABEL_PATH}\n"
            f"请先运行 src/03_train_model.py"
        )

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    model = joblib.load(MODEL_PATH)
    label_mapping = joblib.load(LABEL_PATH)

    id_to_label = label_mapping["id_to_label"]

    return df, model, id_to_label


# ==============================
# 4. 根据模型预测拥堵状态
# ==============================

def predict_congestion(df, model, id_to_label):
    """
    输入某个时间点的四方向交通数据，
    输出模型预测的拥堵状态。
    """

    X = df[FEATURE_COLUMNS]

    pred_ids = model.predict(X)

    pred_labels = []
    for pred_id in pred_ids:
        pred_labels.append(id_to_label[int(pred_id)])

    result_df = df.copy()
    result_df["predicted_congestion"] = pred_labels

    return result_df


# ==============================
# 5. 基础绿灯时间规则
# ==============================

def get_base_green_time(congestion_level):
    """
    根据拥堵状态给出基础绿灯时间。
    """

    if congestion_level == "畅通":
        return 25
    elif congestion_level == "缓行":
        return 35
    elif congestion_level == "拥堵":
        return 50
    else:
        return 30


# ==============================
# 6. 绿灯时间微调规则
# ==============================

def adjust_green_time(row):
    """
    在基础绿灯时间上，根据交通流参数进行微调。
    """

    congestion_level = row["predicted_congestion"]

    green_time = get_base_green_time(congestion_level)

    # 排队长度较长，增加绿灯时间
    if row["queue_length"] > 35:
        green_time += 5

    # 车流量较大，增加绿灯时间
    if row["vehicle_count"] > 110:
        green_time += 5

    # 平均速度很低，说明通行效率差，增加绿灯时间
    if row["avg_speed"] < 20:
        green_time += 5

    # 如果非常畅通，可以略微减少绿灯时间
    if (
        congestion_level == "畅通"
        and row["vehicle_count"] < 30
        and row["queue_length"] < 5
    ):
        green_time -= 5

    # 限制绿灯时间范围
    green_time = max(20, min(60, green_time))

    return int(green_time)


# ==============================
# 7. 生成建议原因
# ==============================

def generate_reason(row):
    """
    生成人类可读的配时建议原因。
    """

    level = row["predicted_congestion"]
    vehicle_count = row["vehicle_count"]
    avg_speed = row["avg_speed"]
    queue_length = row["queue_length"]

    if level == "拥堵":
        reason = (
            f"当前方向预测为拥堵，车流量为{vehicle_count:.1f}，"
            f"平均速度为{avg_speed:.1f}km/h，排队长度为{queue_length:.1f}，"
            f"建议延长绿灯时间以提高通行能力。"
        )
    elif level == "缓行":
        reason = (
            f"当前方向预测为缓行，交通压力中等，"
            f"建议给予适中的绿灯时间。"
        )
    else:
        reason = (
            f"当前方向预测为畅通，交通压力较小，"
            f"可适当缩短绿灯时间，将更多通行时间分配给拥堵方向。"
        )

    return reason


# ==============================
# 8. 生成信号灯配时建议
# ==============================

def generate_signal_timing_plan(predicted_df):
    """
    根据预测结果生成四方向信号灯配时建议。
    """

    result_df = predicted_df.copy()

    result_df["suggested_green_time"] = result_df.apply(
        adjust_green_time,
        axis=1
    )

    result_df["suggestion_reason"] = result_df.apply(
        generate_reason,
        axis=1
    )

    display_columns = [
        "timestamp",
        "direction",
        "vehicle_count",
        "avg_speed",
        "queue_length",
        "occupancy",
        "predicted_congestion",
        "suggested_green_time",
        "suggestion_reason"
    ]

    return result_df[display_columns]


# ==============================
# 9. 选择一个示例时间点
# ==============================

def select_sample_intersection_data(df):
    """
    选择一个早高峰时间点的四方向数据作为示例。
    """

    sample_df = df[
        (df["hour"] == 8) &
        (df["minute"] == 0)
    ]

    # 取第一个满足条件的时间点
    sample_time = sample_df["timestamp"].iloc[0]

    intersection_df = df[df["timestamp"] == sample_time].copy()

    return intersection_df


# ==============================
# 10. 主函数
# ==============================

def main():
    df, model, id_to_label = load_data_and_model()

    print("数据和模型加载成功！")

    # 选择某一时刻四个方向的数据
    intersection_df = select_sample_intersection_data(df)

    print("\n选取的交叉口时间点：")
    print(intersection_df["timestamp"].iloc[0])

    print("\n原始四方向交通数据：")
    print(intersection_df[[
        "direction",
        "vehicle_count",
        "avg_speed",
        "queue_length",
        "occupancy",
        "is_peak"
    ]])

    # 模型预测拥堵状态
    predicted_df = predict_congestion(
        intersection_df,
        model,
        id_to_label
    )

    # 生成配时建议
    signal_plan = generate_signal_timing_plan(predicted_df)

    print("\n信号灯配时建议：")
    print(signal_plan[[
        "direction",
        "predicted_congestion",
        "suggested_green_time"
    ]])

    # 保存结果
    save_path = os.path.join(OUTPUT_DIR, "09_signal_timing_suggestion.csv")
    signal_plan.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"\n配时建议结果已保存：{save_path}")

    print("\n详细建议原因：")
    for _, row in signal_plan.iterrows():
        print(
            f"{row['direction']}方向："
            f"{row['predicted_congestion']}，"
            f"建议绿灯 {row['suggested_green_time']} 秒。"
        )
        print(row["suggestion_reason"])
        print("-" * 60)

    print("\nPhase 4 完成：信号灯配时建议规则系统运行成功！")


if __name__ == "__main__":
    main()