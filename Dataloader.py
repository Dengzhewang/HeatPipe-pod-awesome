# 但愿感君一回顾 使我思君朝与暮
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
# An=np.genfromtxt("An.csv",delimiter=",")
# An_transpose=np.transpose(An)
# An_transpose=An_transpose[:,:6]
# np.savetxt("dataset_An_transpose.csv",An_transpose,delimiter=",")

class DataNormalizer:
    def __init__(self,x_means,x_stds,y_means,y_stds):
        self.x_means=x_means
        self.x_stds=x_stds
        self.y_means=y_means
        self.y_stds=y_stds

    def transform(self,X_data,Y_data):
        X_normalized=(X_data-self.x_means)/self.x_stds
        Y_normalized=(Y_data-self.y_means)/self.y_stds
        return X_normalized,Y_normalized
    def transform_x(self,X_data):
        X_normalized=(X_data-self.x_means)/self.x_stds
        return X_normalized
    def inverse_transform(self,Y_norm):
        return Y_norm*self.y_stds+self.y_means

class DataNormalizer_minmax:
    def __init__(self,x_max,x_min,y_max,y_min):
        self.x_max=x_max
        self.x_min=x_min
        self.y_max=y_max
        self.y_min=y_min

    def transform(self,X_data,Y_data):
        X_normalized=(X_data-self.x_min)/(self.x_max-self.x_min)
        Y_normalized=(Y_data-self.y_min)/(self.y_max-self.y_min)
        return X_normalized,Y_normalized
    def transform_x(self,X_data):
        X_normalized=(X_data-self.x_min)/(self.x_max-self.x_min)
        return X_normalized
    def inverse_transform(self,Y_norm):
        return (self.y_max-self.y_min)*Y_norm+self.y_min



class DataLine:
    def __init__(self,csv_path,test_size,random_state,batch_size,Z_score):
        self.csv_path=csv_path
        self.test_size=test_size
        self.batch_size=batch_size
        self.x_cols=2
        self.y_cols=6
        self.random_state=random_state
        self.z_score=Z_score

    class DataNormalizer:
        def __init__(self, x_means, x_stds, y_means, y_stds):
            self.x_means = x_means
            self.x_stds = x_stds
            self.y_means = y_means
            self.y_stds = y_stds

        def transform(self, X_data, Y_data):
            X_normalized = (X_data - self.x_means) / self.x_stds
            Y_normalized = (Y_data - self.y_means) / self.y_stds
            return X_normalized, Y_normalized

        def transform_x(self, X_data):
            X_normalized = (X_data - self.x_means) / self.x_stds
            return X_normalized

        def inverse_transform(self, Y_norm):
            return Y_norm * self.y_stds + self.y_means

    class DataNormalizer_minmax:
        def __init__(self, x_max, x_min, y_max, y_min):
            self.x_max = x_max
            self.x_min = x_min
            self.y_max = y_max
            self.y_min = y_min

        def transform(self, X_data, Y_data):
            X_normalized = (X_data - self.x_min) / (self.x_max - self.x_min)
            Y_normalized = (Y_data - self.y_min) / (self.y_max - self.y_min)
            return X_normalized, Y_normalized

        def transform_x(self, X_data):
            X_normalized = (X_data - self.x_min) / (self.x_max - self.x_min)
            return X_normalized

        def inverse_transform(self, Y_norm):
            return (self.y_max - self.y_min) * Y_norm + self.y_min

    def load_and_split(self):
        data=np.loadtxt(self.csv_path,delimiter=",",dtype=np.float32)

        input_data=data[:,:self.x_cols]
        output_data=data[:,2:]

        X_train,X_test,y_train,y_test=train_test_split(
            input_data,output_data,
            test_size=self.test_size,
            random_state=self.random_state
        )
        if self.z_score:
            X_means = X_train.mean(axis=0)
            X_stds=X_train.std(axis=0, ddof=1)
            Y_means=y_train.mean(axis=0)
            Y_stds=y_train.std(axis=0,ddof=1)
            normalizer = DataNormalizer(x_means=X_means,x_stds=X_stds,
                                        y_means=Y_means,y_stds=Y_stds)
            X_train_normalized,y_train_normalized= normalizer.transform(X_train,y_train)
            X_test_normlized,y_test_normaliezd=normalizer.transform(X_test,y_test)
            return X_means, X_stds, Y_means, Y_stds, X_train_normalized, X_test_normlized, y_train_normalized, y_test_normaliezd
        else:
            X_min = X_train.min(axis=0)
            X_max = X_train.max(axis=0)
            Y_min = y_train.min(axis=0)
            Y_max = y_train.max(axis=0)
            normalizer = DataNormalizer_minmax(x_max=X_max, x_min=X_min, y_max=Y_max, y_min=Y_min)
            X_train_normlized, y_train_normalized = normalizer.transform(X_data=X_train, Y_data=y_train)
            X_test_normlized, y_test_normaliezd = normalizer.transform(X_data=X_test, Y_data=y_test)

            return X_min, X_max, Y_min, Y_max, X_train_normlized, X_test_normlized, y_train_normalized, y_test_normaliezd


    def creat_dataloaders(self):
        X_min,X_max,Y_min,Y_max,X_train,X_test,y_train,y_test=self.load_and_split()


        X_train_tensor=torch.FloatTensor(X_train)
        y_train_tensor=torch.FloatTensor(y_train)
        X_test_tensor=torch.FloatTensor(X_test)
        y_test_tensor=torch.FloatTensor(y_test)


        train_dataset=TensorDataset(X_train_tensor,y_train_tensor)
        test_dataset=TensorDataset(X_test_tensor,y_test_tensor)


        train_loader=DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        test_loader=DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        return train_loader,test_loader
    def initial(self):
        data = np.loadtxt(self.csv_path, delimiter=",", dtype=np.float32)
        input_data = data[:, :self.x_cols]
        output_data = data[:, 2]

        X_train, X_test, y_train, y_test = train_test_split(
            input_data, output_data,
            test_size=self.test_size,
            random_state=self.random_state
        )
        return y_train,y_test



# dataloader = DataLine(
#         csv_path="uxy_pod_data/an_uxy.csv",
#         test_size=0.1,
#          batch_size=1,
#          random_state=12,
#           Z_score=False
#      )
# X_means, X_stds, Y_means, Y_stds, X_train_normalized, X_test_normlized, y_train_normalized, y_test_normaliezd=dataloader.load_and_split()
# train_loader, test_loader = dataloader.creat_dataloaders()
# print(X_means,X_stds,Y_means,Y_stds)
# y_train,y_test=dataloader.initial()
# print(y_train.shape,y_test.shape)

# np.savetxt("coeffs_predict/pod6_test_inital.csv",y_test,delimiter=",")
# print(train_loader.shape)
# normalizer = DataNormalizer_minmax(x_max=X_stds, x_min=X_means, y_max=X_stds, y_min=X_means)
# for x,y in test_loader:
#
#     x=normalizer.inverse_transform(np.array(x))
#     print(x)
# np.savetxt("x_train_norm_zscore.csv",X_train_normalized,delimiter=",")
# np.savetxt("y_train_norm_zscore.csv",y_train_normalized,delimiter=",")
# np.savetxt("x_test_norm_zscore.csv",X_test_normlized,delimiter=",")
# np.savetxt("y_test_norm_zscore.csv",y_test_normaliezd,delimiter=",")
