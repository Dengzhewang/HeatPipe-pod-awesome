# 但愿感君一回顾 使我思君朝与暮

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from Dataloader import DataLine
import os
os.environ["OMP_NUM_THREADS"] = "1"


# ==============================
# 数据标准化
# ==============================
# class StandardScaler:
#
#     def fit(self, X):
#         self.mean = X.mean(axis=0)
#         self.std = X.std(axis=0)
#
#     def transform(self, X):
#         return (X - self.mean) / self.std
#
#     def inverse_transform(self, X):
#         return X * self.std + self.mean
#

# ==============================
# RBF神经网络
# ==============================
class RBFNN:

    def __init__(self, n_centers,sigma_scale):

        self.n_centers = n_centers
        self.sigma_scale=sigma_scale

    def _rbf(self, X, centers, sigma):
        """
        计算RBF响应矩阵 Φ
        """

        N = X.shape[0]
        M = centers.shape[0]

        Phi = np.zeros((N, M))

        for i in range(N):
            for j in range(M):

                r = np.linalg.norm(X[i] - centers[j])
                Phi[i, j] = np.exp(-(r**2) / (2 * sigma**2))

        return Phi

    def fit(self, X, Y):

        # 1 K-means 找中心
        kmeans = KMeans(n_clusters=self.n_centers, random_state=0)
        kmeans.fit(X)

        self.centers = kmeans.cluster_centers_

        # 2 自动计算 σ
        dmax = 0
        for i in range(self.n_centers):
            for j in range(self.n_centers):

                d = np.linalg.norm(self.centers[i] - self.centers[j])
                dmax = max(dmax, d)

        self.sigma = self.sigma_scale*dmax / np.sqrt(2 * self.n_centers)

        # 3 构造 Φ
        Phi = self._rbf(X, self.centers, self.sigma)

        # 4 求权重
        self.W = np.linalg.pinv(Phi) @ Y

    def predict(self, X):

        Phi = self._rbf(X, self.centers, self.sigma)

        Y_pred = Phi @ self.W

        return Y_pred

    # def predict(self, X):
    #
    #     X = np.atleast_2d(X)
    #
    #     diff = X[:, None, :] - self.centers[None, :, :]
    #
    #     dist = np.sum(diff ** 2, axis=2)
    #
    #     Phi = np.exp(-dist / (2 * self.sigma ** 2))
    #
    #     y = Phi @ self.W
    #
    #     return y


# ==============================
# 主程序
# ==============================

def main():

    # ==============================
    # 读取数据
    # ==============================

    dataloader = DataLine(
            csv_path="uxy_pod_data/An_uxy.CSV",
            test_size=0.1,
             batch_size=1,
             random_state=12,
              Z_score=True
         )
    X_means, X_stds, Y_means, Y_stds, X_train_normalized, X_test_normlized, y_train_normalized, y_test_normalized=dataloader.load_and_split()
    # X_min, X_max, Y_min, Y_max, X_train_normalized, X_test_normlized, y_train_normalized, y_test_normaliezd=dataloader.load_and_split()
    # ==============================
    # 构建RBF网络
    # ==============================

    # rbf = RBFNN(n_centers=55,sigma_scale=0.8)
    #
    #
    # rbf.fit(X_train_normalized, y_train_normalized)

    # ==============================
    # 预测
    # ==============================

    # Y_pred_norm = rbf.predict(X_test_normlized)
    #
    # Y_pred_norm = rbf.predict(np.array([[1,1]]))
    # print(Y_pred_norm.shape)

    # 反归一化
    # Y_pred = Y_pred_norm*Y_stds+Y_means
    # Y_true = y_test_normalized*Y_stds+Y_means


    # Y_pred=Y_pred*(Y_max-Y_min)+Y_min
    # Y_true=y_test_normaliezd*(Y_max-Y_min)+Y_min
    # ==============================
    # 误差计算
    # ==============================

    # rel_error = np.linalg.norm(Y_true - Y_pred) / np.linalg.norm(Y_true)

    # print("Relative Error:", rel_error)
    # print(f"y_true:{Y_true},y_pred；{Y_pred}")
    # erro=(abs(Y_true-Y_pred)).mean()
    # print(erro)
    # print(Y_pred_norm)
    # print(y_train_normalized)

    # ===============================
    # ==========调参==================
    # ================================

    centers_list = [10, 15, 20, 25, 30,35,40,45,50,55]
    sigma_scales= [0.5,0.8, 1,1.5 ,2,2.5, 3,3.5,4,4.5,5,5.5,6]
    best_score=1
    for c in centers_list:
        for s in sigma_scales:
            rbf = RBFNN(n_centers=c, sigma_scale=s)
            rbf.fit(X_train_normalized, y_train_normalized)

            Y_pred = rbf.predict(X_test_normlized)

            # error = np.linalg.norm(y_test_normalized - Y_pred) / np.linalg.norm(y_test_normalized)
            # error=max(abs(Y_pred-y_test_normalized))
            #
            # print(c, s, error)

            error = np.mean((Y_pred - y_test_normalized) ** 2)

            if error < best_score:
                best_score = error
                best_params = (c, s)

    print("最优参数:", best_params,best_score)
    #    =====================================
    #    ===========模型保存====================
    #    ======================================
    # def save_rbf_model(model,name):
    #     os.makedirs("rbf_models",exist_ok=True)
    #
    #     np.save(f"P_models/rbf_models/{name}_centers.npy",model.centers)
    #     np.save(f"P_models/rbf_models/{name}_sigma.npy",model.sigma)
    #     np.save(f"P_models/rbf_models/{name}_w.npy",model.W)
    #     np.save(f"P_models/rbf_models/{name}_sigma_scale.npy",model.sigma_scale)
    #
    # rbf = RBFNN(n_centers=45, sigma_scale=4.5)
    # rbf.fit(X_train_normalized, y_train_normalized)
    # save_rbf_model(rbf,f"pod1")
    # print(rbf.W.shape)

if __name__ == "__main__":
    main()