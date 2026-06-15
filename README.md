# 基于机器学习的交通拥堵状态识别与信号配时建议系统

## 一、项目简介

本项目面向单个十字交叉口场景，构建一个基于机器学习的交通拥堵状态识别与信号配时建议系统。系统使用模拟交通流数据，提取车流量、平均速度、排队长度、道路占有率、时间特征和方向特征，训练拥堵状态分类模型，并根据预测结果输出信号灯绿灯时间建议。

## 二、项目功能

- 模拟生成交通流数据
- 交通数据可视化分析
- 拥堵状态三分类识别：畅通、缓行、拥堵
- 决策树、随机森林、XGBoost 模型对比
- 随机森林特征重要性分析
- 基于拥堵状态的信号灯配时建议
- Streamlit 可视化网页展示
- 支持单方向预测和四方向配时演示

## 三、项目结构

```text
traffic_congestion_project/
├── app.py
├── data/
│   └── simulated_traffic_data.csv
├── models/
│   ├── rf_model.pkl
│   └── label_mapping.pkl
├── outputs/
├── src/
│   ├── 01_generate_data.py
│   ├── 02_data_visualization.py
│   ├── 03_train_model.py
│   └── 04_signal_timing.py
├── requirements.txt
└── README.md