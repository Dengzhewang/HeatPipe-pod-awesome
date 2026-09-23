# 但愿感君一回顾 使我思君朝与暮


import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# data = np.loadtxt(
#     "RESULT_uxy_predict/pod_best_uxy_predict_17.01KW_1273.15K",
#     delimiter=","
# )
#
# # reshape 成一行
# data_row = data.reshape(1, -1)
#
# # 保存
#
# with open ('RESULT_uxy_predict/uxy_predict_data.csv',"ab") as f:
#     np.savetxt(f, data_row, delimiter=",")
a=input("请输入对象：")
predict_data=np.loadtxt(f"RESULT_{a}_predict/{a}_predict_data.csv",delimiter=",")
true_data=np.loadtxt(f"{a}_pod_data/{a}_true_data.csv",delimiter=",")
# 保证形状一致（很重要！）
# true_data=true_data[:,:800]
if a=="T":
    pass
elif a=="P":
    true_data=true_data[:,:200]
elif a=="U":
    true_data=true_data[:,:200]
elif a=="uxy":
    true_data=true_data[:,:800]

assert predict_data.shape == true_data.shape

# 误差
error = predict_data - true_data

# MAE
mae = np.mean(np.abs(error))

# MSE
mse = np.mean(error**2)

# RMSE
rmse = np.sqrt(mse)

# 最大误差（绝对值）
max_error = np.max(np.abs(error))
# r2 = r2_score(true_data, predict_data)
# r2=r2_score(predict_data,true_data)
ss_res = np.sum((true_data - predict_data)**2)
ss_tot = np.sum((true_data - np.mean(true_data))**2)

# r2 = 1 - ss_res / ss_tot

print(f"R^2 = {r2:.6f}")
print(f"MAE  = {mae}")
print(f"MSE  = {mse}")
print(f"RMSE = {rmse}")
print(f"Max Error = {max_error}")

# np.savetxt("RESULT_uxy_predict/error_data.csv",error,delimiter=",")
