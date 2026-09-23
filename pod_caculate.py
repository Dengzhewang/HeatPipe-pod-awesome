# 但愿感君一回顾 使我思君朝与暮
import numpy as np
import scipy.io as io
def POD(U,r=None):  #输入原始数据矩阵，N条m维数据
    U_mean=np.mean(U,axis=1,keepdims=True)
    U=U-U_mean
    N=U.shape[1]
    C=np.dot(U.T,U)/N
    C_lam,C_phi=np.linalg.eigh(C)
    #由小到大排列
    indices=np.argsort(C_lam)[::-1]
    C_lam=C_lam[indices]
    C_phi=C_phi[:,indices]
    #去除数值误差导致的负特征值
    tol=1e-12
    positive=C_lam>tol
    C_lam=C_lam[positive]
    C_phi=C_phi[:,positive]
    #截断阶数
    if r is not None:
        C_lam=C_lam[:r]
        C_phi=C_phi[:,:r]


    U_phi=U @ C_phi/np.sqrt(C_lam*N)

    An=U_phi.T @ U
    energy_ratio=np.cumsum(C_lam)/np.sum(C_lam)
    return U_mean,C_lam,C_phi,An,U_phi,energy_ratio


# m,n=1000,50
# U=np.random.rand(m,n)
#
# U_mean,C_lam,C_phi,An,U_phi,energy_ratio=POD(U)

# print(C_phi.shape)
# print(An.shape)
# print(U_phi.shape)
# print(energy_ratio)
# U_construct=U_phi @ An+U_mean
# erro_l2=np.linalg.norm(U-U_construct)/np.linalg.norm(U)
# print(erro_l2)
# print(abs(U_construct-U))

# for j in range(An.shape[0]):
#     u=U_phi @ An[:,j+1]
#     print(u.shape)

U=np.genfromtxt('T_pod_data/data_T.csv',delimiter=",")
U=U[:,2:]

U_transpose=np.transpose(U)
# print(U_transpose)
U_mean, C_lam, C_phi, An, U_phi, energy_ratio = POD(U_transpose,r=10)
# print(C_lam)

U_construct = U_phi @ An + U_mean
print(abs(U_construct-U_transpose))
erro_l2=np.linalg.norm(U_transpose-U_construct)/np.linalg.norm(U_transpose)

print(erro_l2)
print(U_construct.shape)
# print(U_phi)
# print(U_phi.shape)
# print(An)
# print(An.shape)
print(energy_ratio)
# print(U_mean)

erro=U_construct-U_transpose
erro_mean=np.mean(np.abs(erro))
erro_max=np.max(np.abs(erro))
erro_min=np.min(np.abs(erro))
print(f"mean:{erro_mean},min:{erro_min},max:{erro_max}")

# np.savetxt("uxy_pod_data/C_lam.csv",C_lam,delimiter=",")
# np.savetxt("uxy_pod_data/energy_ratio",energy_ratio,delimiter=",")
#
# np.savetxt("uxy_pod_data/An.csv",An,delimiter=",")
# np.savetxt("uxy_pod_data/U_phi.csv",U_phi,delimiter=",")
# np.savetxt("uxy_pod_data/U_mean.csv",U_mean,delimiter=",")

