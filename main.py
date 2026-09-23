# import numpy as np
# x=np.genfromtxt('data.csv',delimiter=",")
# x=x[:,1:201]
# x=np.transpose(x)
# print(x.shape)
# x_mean=x.mean(axis=1,keepdims=True)
# xp=x-x_mean
# print(xp)
# U,S,Vt=np.linalg.svd(xp,full_matrices=False)
# xr=x_mean+U@np.diag(S)@Vt
# print(np.linalg.norm(x-xr))
import numpy as np
def POD(U, r=None):
    U_mean = np.mean(U, axis=1, keepdims=True)
    Uc = U - U_mean

    N = Uc.shape[1]
    C = Uc.T @ Uc

    # 对称矩阵必须用 eigh
    lam, V = np.linalg.eigh(C)

    idx = np.argsort(lam)[::-1]
    lam = lam[idx]
    V = V[:, idx]

    tol = 1e-12
    pos = lam > tol
    lam = lam[pos]
    V = V[:, pos]

    if r is not None:
        lam = lam[:r]
        V = V[:, :r]

    Phi = Uc @ V / np.sqrt(lam)
    A = Phi.T @ Uc

    # 正交性自检（强烈建议你加）
    # print(np.linalg.norm(Phi.T @ Phi - np.eye(Phi.shape[1])))

    energy_ratio = np.cumsum(lam) / np.sum(lam)
    print(np.sum(lam), np.linalg.norm(Uc) ** 2 / N)

    return U_mean, lam, Phi, A, energy_ratio


U=np.genfromtxt('data.csv',delimiter=",")
U=U[:,1:201]

U_transpose=np.transpose(U)
U_mean, lam, Phi, A, _ = POD(U_transpose, r=None)
U_rec = U_mean + Phi @ A

print(np.linalg.norm(U_transpose - U_rec))
print(np.linalg.norm(Phi.T @ Phi - np.eye(Phi.shape[1])))


