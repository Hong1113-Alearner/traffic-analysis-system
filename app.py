import os
import joblib

import pandas as pd
import streamlit as st

def get_max_queue_by_speed(avg_speed):
    """
    根据平均速度限制排队长度的最大值。
    速度越高，理论上排队长度越小。
    """
    if avg_speed >= 60:
        return 10
    elif avg_speed >= 45:
        return 20
    elif avg_speed >= 30:
        return 35
    else:
        return 50


def get_max_occupancy_by_speed(avg_speed):
    """
    根据平均速度限制道路占有率的最大值。
    速度越高，道路占有率通常不应过高。
    """
    if avg_speed >= 60:
        return 0.40
    elif avg_speed >= 45:
        return 0.60
    elif avg_speed >= 30:
        return 0.80
    else:
        return 1.00


# ==============================
# 1. 页面基础配置
# ==============================

st.set_page_config(
    page_title="交通拥堵状态识别与信号配时建议系统",
    layout="wide"
)


# ==============================
# 2. 路径设置
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "simulated_traffic_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
LABEL_PATH = os.path.join(BASE_DIR, "models", "label_mapping.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


# ==============================
# 3. 模型特征列
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
# 4. 加载数据和模型
# ==============================

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        return None, None

    model = joblib.load(MODEL_PATH)
    label_mapping = joblib.load(LABEL_PATH)

    id_to_label = label_mapping["id_to_label"]

    return model, id_to_label


# ==============================
# 5. 标签转换函数
# ==============================

def decode_label(pred_id, id_to_label):
    """
    将模型预测出的数字标签转换为中文标签。
    """

    pred_id = int(pred_id)

    if pred_id in id_to_label:
        return id_to_label[pred_id]

    if str(pred_id) in id_to_label:
        return id_to_label[str(pred_id)]

    return "未知"


# ==============================
# 6. 信号灯配时规则
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


def adjust_green_time(row):
    """
    在基础绿灯时间上，根据车流量、速度、排队长度进行微调。
    """

    congestion_level = row["predicted_congestion"]

    green_time = get_base_green_time(congestion_level)

    if row["queue_length"] > 35:
        green_time += 5

    if row["vehicle_count"] > 110:
        green_time += 5

    if row["avg_speed"] < 20:
        green_time += 5

    if (
        congestion_level == "畅通"
        and row["vehicle_count"] < 30
        and row["queue_length"] < 5
    ):
        green_time -= 5

    green_time = max(20, min(60, green_time))

    return int(green_time)


def generate_reason(row):
    """
    生成配时建议原因。
    """

    level = row["predicted_congestion"]

    if level == "拥堵":
        return "当前方向交通压力较大，建议延长绿灯时间，提高车辆通行能力。"
    elif level == "缓行":
        return "当前方向交通压力中等，建议给予适中的绿灯时间。"
    else:
        return "当前方向交通较为畅通，可适当缩短绿灯时间。"


def predict_and_generate_plan(input_df, model, id_to_label):
    """
    输入交通流数据，输出拥堵预测结果和绿灯配时建议。
    """

    X = input_df[FEATURE_COLUMNS]

    pred_ids = model.predict(X)

    result_df = input_df.copy()
    result_df["predicted_congestion"] = [
        decode_label(pred_id, id_to_label) for pred_id in pred_ids
    ]

    result_df["suggested_green_time"] = result_df.apply(
        adjust_green_time,
        axis=1
    )

    result_df["suggestion_reason"] = result_df.apply(
        generate_reason,
        axis=1
    )

    return result_df


# ==============================
# 7. 加载资源
# ==============================

df = load_data()
model, id_to_label = load_model_and_labels()


# ==============================
# 8. 页面标题
# ==============================

st.title("基于机器学习的交通拥堵状态识别与信号配时建议系统")

st.markdown(
    """
    本系统基于模拟交通流数据，使用机器学习模型识别交通拥堵状态，
    并根据预测结果输出信号灯绿灯时间建议。
    """
)


# ==============================
# 9. 侧边栏导航
# ==============================

st.sidebar.title("功能菜单")

page = st.sidebar.radio(
    "请选择页面",
    [
        "项目介绍",
        "交通数据分析",
        "模型结果展示",
        "单方向手动预测",
        "四方向配时演示"
    ]
)


# ==============================
# 10. 页面一：项目介绍
# ==============================

if page == "项目介绍":
    st.header("项目介绍")

    st.subheader("项目目标")
    st.write(
        """
        本项目面向单个十字交叉口场景，构建一个基于机器学习的交通拥堵状态识别
        与信号灯配时建议系统。系统通过分析车流量、平均速度、排队长度、道路占有率
        等交通流特征，预测当前交通状态，并输出对应的绿灯时间建议。
        """
    )

    st.subheader("系统技术路线")

    st.code(
        """
模拟交通流数据
        ↓
数据处理与特征工程
        ↓
拥堵状态分类模型
        ↓
模型预测：畅通 / 缓行 / 拥堵
        ↓
配时建议规则系统
        ↓
可视化展示
        """,
        language="text"
    )

    st.subheader("系统模块")
    module_df = pd.DataFrame({
        "模块": [
            "数据处理与特征工程",
            "拥堵状态分类模型",
            "配时建议规则系统",
            "可视化展示"
        ],
        "说明": [
            "构造交通流数据，并提取车流量、速度、排队长度等特征",
            "使用随机森林模型预测交通状态",
            "根据预测结果输出建议绿灯时间",
            "通过网页展示数据分析、模型结果和预测功能"
        ]
    })

    st.table(module_df)


# ==============================
# 11. 页面二：交通数据分析
# ==============================

elif page == "交通数据分析":
    st.header("交通数据分析")

    if df is None:
        st.error("未找到数据文件，请先运行 src/01_generate_data.py")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("数据总量", len(df))

    with col2:
        st.metric("方向数量", df["direction"].nunique())

    with col3:
        st.metric("开始时间", str(df["timestamp"].min())[:10])

    with col4:
        st.metric("结束时间", str(df["timestamp"].max())[:10])

    st.subheader("数据预览")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("拥堵状态分布")
    congestion_counts = df["congestion_level"].value_counts()
    st.bar_chart(congestion_counts)

    st.subheader("一天内不同小时的平均车流量变化")
    hourly_flow = df.groupby("hour")["vehicle_count"].mean()
    st.line_chart(hourly_flow)

    st.subheader("不同方向平均车流量对比")
    direction_flow = df.groupby("direction")["vehicle_count"].mean()
    st.bar_chart(direction_flow)

    st.subheader("不同拥堵状态下的平均速度")
    speed_by_level = df.groupby("congestion_level")["avg_speed"].mean()
    speed_by_level = speed_by_level.reindex(["畅通", "缓行", "拥堵"])
    st.bar_chart(speed_by_level)

    st.info(
        "从这些图表可以观察数据是否符合交通规律，例如早晚高峰车流量更高、拥堵状态下平均速度更低。"
    )


# ==============================
# 12. 页面三：模型结果展示
# ==============================

elif page == "模型结果展示":
    st.header("模型结果展示")

    st.write(
        """
        本项目使用决策树、随机森林和 XGBoost 进行模型对比，
        最终选择随机森林作为主要拥堵状态分类模型。
        """
    )

    image_files = {
        "不同模型准确率对比": "06_model_accuracy_comparison.png",
        "随机森林混淆矩阵": "07_rf_confusion_matrix.png",
        "随机森林特征重要性": "08_rf_feature_importance.png"
    }

    for title, file_name in image_files.items():
        image_path = os.path.join(OUTPUT_DIR, file_name)

        st.subheader(title)

        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning(
                f"未找到 {file_name}，请先运行 src/03_train_model.py"
            )

    st.subheader("模型说明")

    st.markdown(
        """
        - **决策树**：作为基础模型，便于理解分类规则。
        - **随机森林**：作为主模型，通过多棵决策树投票，提高分类稳定性。
        - **XGBoost**：作为对比模型，用于验证集成学习方法的分类效果。
        """
    )


# ==============================
# 13. 页面四：单方向手动预测
# ==============================

elif page == "单方向手动预测":
    st.header("单方向手动预测")

    if model is None or id_to_label is None:
        st.error("未找到模型文件，请先运行 src/03_train_model.py")
        st.stop()

    st.write(
        """
        在该页面中，可以手动输入某一方向的交通流参数，
        系统将预测该方向的拥堵状态，并给出建议绿灯时间。
        """
    )

    direction_display = {
        "东进口": "East",
        "南进口": "South",
        "西进口": "West",
        "北进口": "North"
    }

    col1, col2 = st.columns(2)

    # 初始化单方向预测页面的滑块状态
    if "single_avg_speed" not in st.session_state:
        st.session_state["single_avg_speed"] = 35

    if "single_queue_length" not in st.session_state:
        st.session_state["single_queue_length"] = 20

    if "single_occupancy" not in st.session_state:
        st.session_state["single_occupancy"] = 0.55

    with col1:
        direction_cn = st.selectbox(
            "进口方向",
            list(direction_display.keys())
        )

        vehicle_count = st.slider(
            "5分钟车流量",
            min_value=0,
            max_value=150,
            value=80
        )

        avg_speed = st.slider(
            "平均速度 km/h",
            min_value=0,
            max_value=80,
            step=1,
            key="single_avg_speed"
        )

        # 根据平均速度动态限制排队长度最大值
        max_queue = get_max_queue_by_speed(avg_speed)

        # 如果之前的排队长度超过新的最大值，自动压回合理范围
        if st.session_state["single_queue_length"] > max_queue:
            st.session_state["single_queue_length"] = max_queue

        queue_length = st.slider(
            "排队长度",
            min_value=0,
            max_value=max_queue,
            step=1,
            key="single_queue_length"
        )

        st.caption(
            f"当前平均速度为 {avg_speed} km/h，排队长度最大值限制为 {max_queue}。"
        )

    with col2:
        # 根据平均速度动态限制道路占有率最大值
        max_occupancy = get_max_occupancy_by_speed(avg_speed)

        # 如果之前的占有率超过新的最大值，自动压回合理范围
        if st.session_state["single_occupancy"] > max_occupancy:
            st.session_state["single_occupancy"] = max_occupancy

        occupancy = st.slider(
            "道路占有率",
            min_value=0.0,
            max_value=max_occupancy,
            step=0.01,
            key="single_occupancy"
        )

        st.caption(
            f"当前平均速度为 {avg_speed} km/h，道路占有率最大值限制为 {max_occupancy:.2f}。"
        )

        hour = st.slider(
            "当前小时",
            min_value=0,
            max_value=23,
            value=8
        )

        weekday = st.selectbox(
            "星期",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: [
                "周一", "周二", "周三", "周四", "周五", "周六", "周日"
            ][x]
        )

        is_weekend = 1 if weekday >= 5 else 0
        is_peak = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0

        st.write("是否周末：", "是" if is_weekend == 1 else "否")
        st.write("是否高峰期：", "是" if is_peak == 1 else "否")

    input_df = pd.DataFrame([{
        "vehicle_count": vehicle_count,
        "avg_speed": avg_speed,
        "queue_length": queue_length,
        "occupancy": occupancy,
        "hour": hour,
        "weekday": weekday,
        "is_weekend": is_weekend,
        "is_peak": is_peak,
        "direction": direction_display[direction_cn]
    }])

    if st.button("开始预测"):
        result_df = predict_and_generate_plan(
            input_df,
            model,
            id_to_label
        )

        result = result_df.iloc[0]

        st.subheader("预测结果")

        col_a, col_b = st.columns(2)

        with col_a:
            st.metric(
                "预测拥堵状态",
                result["predicted_congestion"]
            )

        with col_b:
            st.metric(
                "建议绿灯时间",
                f"{result['suggested_green_time']} 秒"
            )

        st.subheader("建议原因")
        st.write(result["suggestion_reason"])

        st.subheader("输入数据")
        display_df = input_df.copy()
        display_df["direction_cn"] = direction_cn
        st.dataframe(display_df, use_container_width=True)


# ==============================
# 14. 页面五：四方向配时演示
# ==============================

elif page == "四方向配时演示":
    st.header("四方向配时演示")

    if model is None or id_to_label is None:
        st.error("未找到模型文件，请先运行 src/03_train_model.py")
        st.stop()

    st.write(
        """
        本页面模拟一个十字交叉口四个进口方向的交通状态，
        系统将分别预测各方向拥堵状态，并输出绿灯时间建议。
        """
    )

    st.subheader("统一时间参数")

    col_time1, col_time2 = st.columns(2)

    with col_time1:
        hour = st.slider(
            "当前小时",
            min_value=0,
            max_value=23,
            value=8,
            key="four_hour"
        )

    with col_time2:
        weekday = st.selectbox(
            "星期",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: [
                "周一", "周二", "周三", "周四", "周五", "周六", "周日"
            ][x],
            key="four_weekday"
        )

    is_weekend = 1 if weekday >= 5 else 0
    is_peak = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0

    st.write("是否高峰期：", "是" if is_peak == 1 else "否")

    st.subheader("四方向交通参数输入")

    direction_info = [
        ("东进口", "East", 100, 28, 25, 0.65),
        ("南进口", "South", 75, 38, 15, 0.50),
        ("西进口", "West", 110, 24, 30, 0.70),
        ("北进口", "North", 65, 45, 10, 0.40)
    ]

    rows = []

    for direction_cn, direction_en, default_flow, default_speed, default_queue, default_occ in direction_info:
        with st.expander(direction_cn, expanded=True):
            col1, col2, col3, col4 = st.columns(4)

            flow_key = f"{direction_en}_flow"
            speed_key = f"{direction_en}_speed"
            queue_key = f"{direction_en}_queue"
            occ_key = f"{direction_en}_occ"

            # 初始化四方向页面每个方向的滑块状态
            if flow_key not in st.session_state:
                st.session_state[flow_key] = default_flow

            if speed_key not in st.session_state:
                st.session_state[speed_key] = default_speed

            if queue_key not in st.session_state:
                st.session_state[queue_key] = default_queue

            if occ_key not in st.session_state:
                st.session_state[occ_key] = default_occ

            with col1:
                vehicle_count = st.slider(
                    f"{direction_cn} - 5分钟车流量",
                    min_value=0,
                    max_value=150,
                    step=1,
                    key=flow_key
                )

            with col2:
                avg_speed = st.slider(
                    f"{direction_cn} - 平均速度",
                    min_value=0,
                    max_value=80,
                    step=1,
                    key=speed_key
                )

            # 根据当前方向的平均速度，动态限制排队长度和道路占有率
            max_queue = get_max_queue_by_speed(avg_speed)
            max_occupancy = get_max_occupancy_by_speed(avg_speed)

            if st.session_state[queue_key] > max_queue:
                st.session_state[queue_key] = max_queue

            if st.session_state[occ_key] > max_occupancy:
                st.session_state[occ_key] = max_occupancy

            with col3:
                queue_length = st.slider(
                    f"{direction_cn} - 排队长度",
                    min_value=0,
                    max_value=max_queue,
                    step=1,
                    key=queue_key
                )

                st.caption(f"上限：{max_queue}")

            with col4:
                occupancy = st.slider(
                    f"{direction_cn} - 道路占有率",
                    min_value=0.0,
                    max_value=max_occupancy,
                    step=0.01,
                    key=occ_key
                )

                st.caption(f"上限：{max_occupancy:.2f}")

            rows.append({
                "direction_cn": direction_cn,
                "direction": direction_en,
                "vehicle_count": vehicle_count,
                "avg_speed": avg_speed,
                "queue_length": queue_length,
                "occupancy": occupancy,
                "hour": hour,
                "weekday": weekday,
                "is_weekend": is_weekend,
                "is_peak": is_peak
            })

    input_df = pd.DataFrame(rows)

    if st.button("生成四方向配时建议"):
        result_df = predict_and_generate_plan(
            input_df,
            model,
            id_to_label
        )

        display_columns = [
            "direction_cn",
            "vehicle_count",
            "avg_speed",
            "queue_length",
            "occupancy",
            "predicted_congestion",
            "suggested_green_time",
            "suggestion_reason"
        ]

        st.subheader("四方向预测与配时建议")
        st.dataframe(
            result_df[display_columns],
            use_container_width=True
        )

        st.subheader("四方向建议绿灯时间对比")
        chart_df = result_df.set_index("direction_cn")["suggested_green_time"]
        st.bar_chart(chart_df)

        st.subheader("展示说明")

        st.write(
            """
            从结果可以看出，系统会根据不同方向的交通流参数预测拥堵状态。
            对于预测为拥堵的方向，系统会给出更长的建议绿灯时间；
            对于畅通方向，系统会给出较短的绿灯时间。
            """
        )