# 但愿感君一回顾 使我思君朝与暮
import torch
import torch.nn as nn
import numpy as np
from Dataloader import DataLine
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import time

class BPNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            # nn.Linear(2,32),
            # nn.LayerNorm(32),
            # # nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Dropout(0.2),
            #
            # nn.Linear(32,64),
            # nn.LayerNorm(64),
            # # nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Dropout(0.2),
            #
            #
            # nn.Linear(64,64),
            # nn.LayerNorm(64),
            # # nn.ReLU(),
            # nn.LeakyReLU(),
            # nn.Dropout(0.2),
            #
            # nn.Linear(64,32),
            # nn.LayerNorm(32),
            # nn.LeakyReLU(),
            # nn.Dropout(0.2),
            #
            # nn.Linear(32,16),
            # nn.LayerNorm(16),
            # nn.LeakyReLU(),
            #
            # nn.Linear(16,6),
            nn.Linear(2,16),
            nn.LayerNorm(16),
            nn.ReLU(),

            nn.Linear(16,16),
            nn.LayerNorm(16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.ReLU(),

            nn.Linear(8,1),

        )
    def forward(self,input):
        return self.net(input)

def train_one_epoch(model,train_loader,loss_fun,optimizer):
    model.train()

    total_mseloss=0
    for input,targert in train_loader:
        optimizer.zero_grad()
        output=model(input)
        mseloss=loss_fun(output,targert)
        mseloss.backward()
        optimizer.step()
        total_mseloss+=mseloss.item()
    avg_mseloss=total_mseloss/len(train_loader)
    return avg_mseloss
def evaluate(model,test_loader,loss_fun):
    model.eval()

    total_mseloss=0
    with torch.no_grad():
        for input,target in test_loader:
            output=model(input)
            mseloss=loss_fun(output,target)
            total_mseloss+=mseloss.item()
        avg_mseloss=total_mseloss/len(test_loader)
        return avg_mseloss

def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
        set_seed(13)

        dataloader=DataLine(
            csv_path="uxy_pod_data/An_uxy.CSV",
            test_size=0.1,
            batch_size=8,
            random_state=12,
            Z_score=False
        )
        X_means, X_stds, Y_means, Y_stds, X_train_normalized, X_test_normlized, y_train_normalized, y_test_normaliezd = dataloader.load_and_split()
        train_loader, test_loader = dataloader.creat_dataloaders()

        model=BPNet()

        loss_fun=nn.MSELoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-3
        )
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
        scheduler = StepLR(
            optimizer,
            step_size=300,
            gamma=0.5
        )
        writer = SummaryWriter("runs/uxy_runs/pod8")

        num_epochs=2000
        for epoch in range(num_epochs):
            start_time = time.time()

            train_mse_loss= train_one_epoch(
                model,
                train_loader,
                loss_fun,
                optimizer,
            )

            val_mse_loss= evaluate(
                model,
                test_loader,
                loss_fun,
            )

            scheduler.step()
            # scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]

            end_time = time.time()

            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train mseLoss: {train_mse_loss:.8f}")

            print(f"Val mseLoss: {val_mse_loss:.8f}")

            print(f"LR: {current_lr:.6e}")
            print(f"Time: {end_time - start_time:.2f}s")
            print("-----------------------------")

            # ===============================
            # TensorBoard记录
            # ===============================

            writer.add_scalar("Loss/train", train_mse_loss, epoch)
            writer.add_scalar("Loss/val", val_mse_loss, epoch)
            writer.add_scalar("LR", current_lr, epoch)

        writer.close()
        torch.save(model.state_dict(),
                       "POD_bp_best_model/model_uxy/pod8.pth")
        print("model saved!")

        print(X_means,X_stds,Y_means,Y_stds)

if __name__=="__main__":
    main()