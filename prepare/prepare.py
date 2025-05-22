# prepare/prepare.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# 加载 EDA 后处理好的数据
df = pd.read_csv("data/processed_car_data.csv")

# 布尔转整数（安全）
for col in df.select_dtypes(include=["bool"]).columns:
    df[col] = df[col].astype(int)

# 强相关特征（来自热力图分析）
selected_features = [
    "km_driven", "mileage", "engine", "max_power", "seats",
    "name_target_enc", "fuel_Diesel", "fuel_Petrol",
    "seller_type_Individual", "transmission_Manual"
]

# 特征 X 和目标 y
X = df[selected_features]
y = df["selling_price"]

# 划分训练集与验证集（60% / 40%）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

# 创建输出目录
os.makedirs("prepare", exist_ok=True)

# 保存数据
X.to_csv("prepare/X_full.csv", index=False)
y.to_csv("prepare/y_full.csv", index=False)
X_train.to_csv("prepare/X_train.csv", index=False)
y_train.to_csv("prepare/y_train.csv", index=False)
X_val.to_csv("prepare/X_val.csv", index=False)
y_val.to_csv("prepare/y_val.csv", index=False)

print("✅ 数据集保存成功X/y 训练验证集就绪！")
