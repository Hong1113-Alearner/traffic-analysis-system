import os

import pandas as pd
import matplotlib.pyplot as plt


# ==============================
# 1. 路径设置
# ==============================

# 当前文件路径：src/02_data_visualization.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录：traffic_congestion_project
BASE_DIR = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "simulated_traffic_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# 2. 解决中文显示问题
# ==============================

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ==============================
# 3. 读取数据
# ==============================

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"没有找到数据文件：{DATA_PATH}\n"
            f"请先运行 src/01_generate_data.py 生成 simulated_traffic_data.csv"
        )

    df = pd.read_csv(DATA_PATH)

    # 把 timestamp 转换成时间类型，方便后面按时间分析
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


# ==============================
# 4. 基础数据检查
# ==============================

def basic_check(df):
    print("=" * 50)
    print("数据基本信息")
    print("=" * 50)

    print("\n数据规模：")
    print(df.shape)

    print("\n前5行数据：")
    print(df.head())

    print("\n字段信息：")
    print(df.info())

    print("\n数值字段统计：")
    print(df.describe())

    print("\n拥堵状态分布：")
    print(df["congestion_level"].value_counts())

    print("\n方向分布：")
    print(df["direction"].value_counts())


# ==============================
# 5. 图1：拥堵状态分布
# ==============================

def plot_congestion_distribution(df):
    congestion_counts = df["congestion_level"].value_counts()

    plt.figure(figsize=(7, 5))
    congestion_counts.plot(kind="bar")

    plt.title("拥堵状态分布")
    plt.xlabel("拥堵状态")
    plt.ylabel("样本数量")
    plt.xticks(rotation=0)

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "01_congestion_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"图1已保存：{save_path}")


# ==============================
# 6. 图2：不同小时平均车流量变化
# ==============================

def plot_hourly_vehicle_count(df):
    hourly_flow = df.groupby("hour")["vehicle_count"].mean()

    plt.figure(figsize=(9, 5))
    plt.plot(hourly_flow.index, hourly_flow.values, marker="o")

    plt.title("一天内不同小时的平均车流量变化")
    plt.xlabel("小时")
    plt.ylabel("平均车流量")
    plt.xticks(range(0, 24))

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "02_hourly_vehicle_count.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"图2已保存：{save_path}")


# ==============================
# 7. 图3：四个方向平均车流量对比
# ==============================

def plot_direction_vehicle_count(df):
    direction_flow = df.groupby("direction")["vehicle_count"].mean()

    plt.figure(figsize=(7, 5))
    direction_flow.plot(kind="bar")

    plt.title("不同进口方向的平均车流量对比")
    plt.xlabel("进口方向")
    plt.ylabel("平均车流量")
    plt.xticks(rotation=0)

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "03_direction_vehicle_count.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"图3已保存：{save_path}")


# ==============================
# 8. 图4：不同拥堵状态下的平均速度
# ==============================

def plot_speed_by_congestion(df):
    speed_by_level = df.groupby("congestion_level")["avg_speed"].mean()

    # 按照交通状态顺序排列
    order = ["畅通", "缓行", "拥堵"]
    speed_by_level = speed_by_level.reindex(order)

    plt.figure(figsize=(7, 5))
    speed_by_level.plot(kind="bar")

    plt.title("不同拥堵状态下的平均速度")
    plt.xlabel("拥堵状态")
    plt.ylabel("平均速度 km/h")
    plt.xticks(rotation=0)

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "04_speed_by_congestion.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"图4已保存：{save_path}")


# ==============================
# 9. 图5：高峰期与非高峰期车流量对比
# ==============================

def plot_peak_vs_nonpeak(df):
    peak_flow = df.groupby("is_peak")["vehicle_count"].mean()

    # 修改索引名称，方便显示
    peak_flow.index = ["非高峰期", "高峰期"]

    plt.figure(figsize=(7, 5))
    peak_flow.plot(kind="bar")

    plt.title("高峰期与非高峰期平均车流量对比")
    plt.xlabel("时间类型")
    plt.ylabel("平均车流量")
    plt.xticks(rotation=0)

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "05_peak_vs_nonpeak.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"图5已保存：{save_path}")


# ==============================
# 10. 主函数
# ==============================

def main():
    df = load_data()

    basic_check(df)

    plot_congestion_distribution(df)
    plot_hourly_vehicle_count(df)
    plot_direction_vehicle_count(df)
    plot_speed_by_congestion(df)
    plot_peak_vs_nonpeak(df)

    print("\n所有可视化图表生成完成！")
    print(f"图片保存位置：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()