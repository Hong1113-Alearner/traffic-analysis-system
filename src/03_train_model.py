import os
import joblib

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ==============================
# 1. 路径设置
# ==============================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "simulated_traffic_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# 2. 中文显示设置
# ==============================

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ==============================
# 3. 读取数据
# ==============================

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"未找到数据文件：{DATA_PATH}\n"
            f"请先运行 src/01_generate_data.py"
        )

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


# ==============================
# 4. 特征工程
# ==============================

def prepare_features(df):
    """
    构造模型输入特征 X 和标签 y
    """

    # 输入特征
    feature_columns = [
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

    X = df[feature_columns]

    # 标签映射：把中文标签转换成数字
    label_to_id = {
        "畅通": 0,
        "缓行": 1,
        "拥堵": 2
    }

    id_to_label = {
        0: "畅通",
        1: "缓行",
        2: "拥堵"
    }

    y = df["congestion_level"].map(label_to_id)

    return X, y, label_to_id, id_to_label


# ==============================
# 5. 构造预处理器
# ==============================

def build_preprocessor():
    """
    对数值特征和类别特征做不同处理。

    数值特征：直接输入模型
    类别特征 direction：做 One-Hot 编码
    """

    numeric_features = [
        "vehicle_count",
        "avg_speed",
        "queue_length",
        "occupancy",
        "hour",
        "weekday",
        "is_weekend",
        "is_peak"
    ]

    categorical_features = [
        "direction"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return preprocessor


# ==============================
# 6. 构造模型流水线
# ==============================

def build_model_pipeline(model):
    """
    Pipeline = 特征预处理 + 模型

    好处：
    1. 训练时自动处理 direction
    2. 预测时也自动处理 direction
    3. 后面保存模型更方便
    """

    preprocessor = build_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


# ==============================
# 7. 训练多个模型
# ==============================

def train_models(X_train, y_train):
    models = {}

    # 1. 决策树
    dt_model = DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    )
    models["Decision Tree"] = build_model_pipeline(dt_model)

    # 2. 随机森林：本项目主模型
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
    models["Random Forest"] = build_model_pipeline(rf_model)

    # 3. XGBoost：进阶对比模型
    if XGBOOST_AVAILABLE:
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss"
        )
        models["XGBoost"] = build_model_pipeline(xgb_model)
    else:
        print("没有安装 xgboost，跳过 XGBoost 模型。")

    trained_models = {}

    for model_name, model_pipeline in models.items():
        print(f"\n正在训练模型：{model_name}")
        model_pipeline.fit(X_train, y_train)
        trained_models[model_name] = model_pipeline
        print(f"{model_name} 训练完成。")

    return trained_models


# ==============================
# 8. 模型评估
# ==============================

def evaluate_models(trained_models, X_test, y_test, id_to_label):
    results = {}

    target_names = [id_to_label[0], id_to_label[1], id_to_label[2]]

    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)

    for model_name, model_pipeline in trained_models.items():
        y_pred = model_pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy

        print(f"\n模型：{model_name}")
        print(f"准确率 Accuracy：{accuracy:.4f}")

        print("\n分类报告：")
        print(classification_report(
            y_test,
            y_pred,
            target_names=target_names
        ))

    return results


# ==============================
# 9. 绘制模型准确率对比图
# ==============================

def plot_model_accuracy(results):
    result_series = pd.Series(results).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    result_series.plot(kind="bar")

    plt.title("不同模型准确率对比")
    plt.xlabel("模型")
    plt.ylabel("准确率")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)

    for index, value in enumerate(result_series.values):
        plt.text(index, value + 0.01, f"{value:.3f}", ha="center")

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "06_model_accuracy_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"模型准确率对比图已保存：{save_path}")


# ==============================
# 10. 绘制随机森林混淆矩阵
# ==============================

def plot_confusion_matrix_for_rf(rf_model, X_test, y_test, id_to_label):
    y_pred = rf_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    labels = [id_to_label[0], id_to_label[1], id_to_label[2]]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("随机森林模型混淆矩阵")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.colorbar()

    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "07_rf_confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"随机森林混淆矩阵已保存：{save_path}")


# ==============================
# 11. 绘制随机森林特征重要性
# ==============================

def plot_feature_importance(rf_model):
    """
    随机森林可以输出每个特征的重要性。
    这对报告和答辩非常有用。
    """

    preprocessor = rf_model.named_steps["preprocess"]
    model = rf_model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=True
    )

    plt.figure(figsize=(8, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])

    plt.title("随机森林特征重要性")
    plt.xlabel("重要性")
    plt.ylabel("特征")

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "08_rf_feature_importance.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"随机森林特征重要性图已保存：{save_path}")


# ==============================
# 12. 保存模型
# ==============================

def save_model(rf_model, label_to_id, id_to_label):
    model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    label_path = os.path.join(MODEL_DIR, "label_mapping.pkl")

    joblib.dump(rf_model, model_path)

    label_mapping = {
        "label_to_id": label_to_id,
        "id_to_label": id_to_label
    }
    joblib.dump(label_mapping, label_path)

    print("\n模型保存完成：")
    print(model_path)
    print(label_path)


# ==============================
# 13. 主函数
# ==============================

def main():
    df = load_data()

    print("数据读取成功！")
    print("数据规模：", df.shape)

    X, y, label_to_id, id_to_label = prepare_features(df)

    print("\n特征列：")
    print(X.columns.tolist())

    print("\n标签分布：")
    print(y.value_counts().sort_index())

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\n训练集规模：", X_train.shape)
    print("测试集规模：", X_test.shape)

    # 训练模型
    trained_models = train_models(X_train, y_train)

    # 评估模型
    results = evaluate_models(
        trained_models,
        X_test,
        y_test,
        id_to_label
    )

    # 绘制模型准确率对比
    plot_model_accuracy(results)

    # 取随机森林作为主模型
    rf_model = trained_models["Random Forest"]

    # 绘制随机森林混淆矩阵
    plot_confusion_matrix_for_rf(
        rf_model,
        X_test,
        y_test,
        id_to_label
    )

    # 绘制随机森林特征重要性
    plot_feature_importance(rf_model)

    # 保存随机森林模型
    save_model(rf_model, label_to_id, id_to_label)

    print("\nPhase 3 完成：模型训练、评估和保存全部完成！")


if __name__ == "__main__":
    main()