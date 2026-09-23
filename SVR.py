# 但愿感君一回顾 使我思君朝与暮
# SVR(支持向量回归) 示例：2个输入 -> 5个输出
# 说明（尽量直白）：
# - SVR 本质是“回归版的SVM”，常用 RBF 核把输入映射到更高维空间，再做线性回归
# - ε (epsilon) 的意思：允许预测值和真实值之间有一段“误差不惩罚区间”（ε-不敏感损失）
# - C 控制“要不要强行拟合训练数据”：
#     C 大：更努力拟合（可能过拟合）
#     C 小：更平滑（可能欠拟合）
# - gamma 控制 RBF 核的“影响范围”：
#     gamma 大：每个样本影响范围小（模型更复杂）
#     gamma 小：更平滑
#
# 多输出怎么做？
# - scikit-learn 的 SVR 默认一次只能输出 1 维
# - 所以我们用 MultiOutputRegressor：内部训练 5 个 SVR，分别预测 a1..a5

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import Dataloader
from sklearn.model_selection import GridSearchCV
import joblib
import os

# -------------------------
# 1) 准备数据（演示用假数据，你要换成自己的）
#    X: (N,2)  输入参数
#    Y: (N,5)  输出（5个POD系数）
# -------------------------

dataloader = Dataloader.DataLine(
        csv_path="P_pod_data/An_data.csv",
        test_size=0.1,
         batch_size=1,
         random_state=12,
          Z_score=True
     )
X_means, X_stds, Y_means, Y_stds, X_train_normalized, X_test_normlized, y_train_normalized, y_test_normaliezd=dataloader.load_and_split()


# -------------------------
# 2) 建立模型（非常推荐加标准化）
#    因为 SVR 对特征尺度很敏感：比如 Re=1e5, AoA=5，会很麻烦
# -------------------------
# Pipeline 的意思是：先做标准化，再做回归（这样不会漏掉预处理）
# svr_single = SVR(
#     kernel="rbf",   # 常用 RBF 核（非线性）
#     C=100.0,         # 拟合强度（可调）
#     epsilon=0.001,   # ε-不惩罚区间（可调）
#     gamma=0.001 # 核宽度参数（"scale" 通常作为默认起点）
# )
#
# model = Pipeline(steps=[
#     ("svr", MultiOutputRegressor(svr_single))
# ])
#
# param_grid = {
#     "svr__estimator__C": [0.1,1,10,100],
#     "svr__estimator__gamma": [0.001,0.01,0.1,1],
#     "svr__estimator__epsilon": [0.001,0.01,0.1]
# }
# grid = GridSearchCV(
#     model,
#     param_grid,
#     cv=5,
#     scoring="neg_mean_squared_error"
# )
#
# grid.fit(X_train_normalized, y_train_normalized)
#
# print("Best params:", grid.best_params_)
# -------------------------
# 3) 训练
# -------------------------
# model.fit(X_train_normalized, y_train_normalized)

# -------------------------
# 4) 测试（看误差）
# -------------------------
# Y_pred_norm = model.predict(X_test_normlized)
#
# datanormalize=Dataloader.DataNormalizer(x_means=np.array([9.894037,1106.884]),x_stds=[5.776581,136.84344 ],
#                       y_means=np.array([2.6540762e+02,-1.1591947e+01,1.0127123e+01,-6.9049358e-02,-1.7301182e-01,-5.0453819e-02]),
#                       y_stds=np.array([5.5085073e+03,3.1386697e+02,2.2012952e+02,3.6421215e+01,1.2449255e+01,4.3632331e+00]))
#
# mse = mean_squared_error(y_test_normaliezd, Y_pred_norm)  # 所有输出维度混在一起的平均MSE
# print("Test MSE =", mse)

# 也可以看每个输出(每个系数)的MSE
# mse_each = ((y_test_normaliezd - Y_pred_norm) ** 2).mean(axis=0)
# print("MSE for each output (a1..a5):", mse_each)

# -------------------------
# 5) 用新输入做预测
# -------------------------
# X_new = np.array([[0.2, -0.4],
#                   [0.7,  0.3]], dtype=np.float32)
#
# Y_new = model.predict(X_new)   # (2,5)
# print("X_new:\n", X_new)
# print("Predicted 5 outputs:\n", Y_new)


# ====================================
# ==============模型保存===============
# ====================================

n_modes=y_train_normalized.shape[1]
param_grid = {
    "C":[0.1,1,10,100,1000],
    "gamma":[0.001,0.01,0.1,1,10],
    "epsilon":[0.001,0.01,0.1]
}
models = []

for i in range(n_modes):

    print(f"Tuning mode {i+1}")

    y_train_i = y_train_normalized[:, i]

    svr = SVR(kernel="rbf")

    grid = GridSearchCV(
        svr,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    grid.fit(X_train_normalized, y_train_i)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    models.append(grid.best_estimator_)



for i, model in enumerate(models):

    joblib.dump(model, f"P_models/svr_models/svr_mode{i+1}.pkl")


# ==================================================
# =======================模型加载====================
# ==================================================

# models = []
#
# for i in range(6):
#
#     model = joblib.load(f"svr_models/svr_mode{i+1}.pkl")
#
#     models.append(model)

# for i in range(6):
#     model1=models[i]
#
#     Y_pred_norm=model1.predict(X_test_normlized)
#     mse = mean_squared_error(y_test_normaliezd[:,[i]], Y_pred_norm)
#     print(f"Test MSE_pod{i+1} =", mse)
#


# datanormalizer=Dataloader.DataNormalizer(x_means=np.array([9.894037,1106.884]),x_stds=[5.776581,136.84344 ],
#                       y_means=np.array([2.6540762e+02,-1.1591947e+01,1.0127123e+01,-6.9049358e-02,-1.7301182e-01,-5.0453819e-02]),
#                       y_stds=np.array([5.5085073e+03,3.1386697e+02,2.2012952e+02,3.6421215e+01,1.2449255e+01,4.3632331e+00]))
#
# x = np.array([13.005,1223.15]).reshape(1,-1)
# x_norm=datanormalizer.transform_x(x)
#
# coeffs = []
#
# for model in models:
#
#     a = model.predict(x_norm)
#
#     coeffs.append(a.item())
#
# coeffs=datanormalizer.inverse_transform(coeffs)
# coeffs=np.array(coeffs).reshape(6,1)
#
# print(coeffs.shape)
# U_phi=np.loadtxt("T_pod_data/U_phi.csv",delimiter=",")
# U_phi=U_phi[:,:6]
# U_mean=np.loadtxt("T_pod_data/U_mean.csv",delimiter=",")
# U_pred=U_mean.reshape(1600,1)+U_phi@coeffs
#
# U_pred=U_pred.reshape(8,200)
# np.savetxt("pod_svr_T_predict_13.005KW_1223.15K",U_pred,delimiter=",")