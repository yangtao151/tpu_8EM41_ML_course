# train_linear.py
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# 加载训练数据
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv")

# 创建模型并训练
model = LinearRegression()
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, "models/linear_model.joblib")
print("✅ 线性回归模型训练完成，已保存为 models/linear_model.joblib")
