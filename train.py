'''
Training function of IEEEVR
'''
import torch
import torch.nn as nn
from data.utils import EdsDataset, train_model, AllEdsDataset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from timm_vit import VisionTransformer
import argparse
import os
from datetime import datetime
from torch.utils.data import random_split, DataLoader
from torchsummary import summary
from torch.nn import DataParallel
import numpy as np
import random


lookup_table = torch.tensor(
    [
        [0.0000e00, 6.7481e00, 6.1958e00],
        [1.7453e-02, 7.1402e00, 6.5931e00],
        [3.4907e-02, 7.5324e00, 6.9905e00],
        [5.2360e-02, 7.9245e00, 7.3879e00],
        [6.9813e-02, 8.3167e00, 7.7853e00],
        [8.7266e-02, 8.7088e00, 8.1826e00],
        [1.0472e-01, 9.1010e00, 8.5800e00],
        [1.2217e-01, 9.4931e00, 8.9774e00],
        [1.3963e-01, 9.8852e00, 9.3748e00],
        [1.5708e-01, 1.0277e01, 9.7721e00],
        [1.7453e-01, 1.0670e01, 1.0170e01],
        [1.9199e-01, 1.1062e01, 1.0567e01],
        [2.0944e-01, 1.1454e01, 1.0964e01],
        [2.2689e-01, 1.1846e01, 1.1362e01],
        [2.4435e-01, 1.2238e01, 1.1759e01],
        [2.6180e-01, 1.2630e01, 1.2156e01],
        [2.7925e-01, 1.3022e01, 1.2554e01],
        [2.9671e-01, 1.3415e01, 1.2951e01],
        [3.1416e-01, 1.3807e01, 1.3349e01],
        [3.3161e-01, 1.4199e01, 1.3746e01],
        [3.4907e-01, 1.4591e01, 1.4143e01],
        [3.6652e-01, 1.4983e01, 1.4541e01],
        [3.8397e-01, 1.5375e01, 1.4938e01],
        [4.0143e-01, 1.5767e01, 1.5335e01],
        [4.1888e-01, 1.6160e01, 1.5733e01],
        [4.3633e-01, 1.6552e01, 1.6130e01],
        [4.5379e-01, 1.6944e01, 1.6528e01],
        [4.7124e-01, 1.7336e01, 1.6925e01],
        [4.8869e-01, 1.7728e01, 1.7322e01],
        [5.0615e-01, 1.8120e01, 1.7720e01],
        [5.2360e-01, 1.8512e01, 1.8117e01],
        [5.4105e-01, 1.8905e01, 1.8514e01],
        [5.5851e-01, 1.9297e01, 1.8912e01],
        [5.7596e-01, 1.9689e01, 1.9309e01],
        [5.9341e-01, 2.0081e01, 1.9707e01],
        [6.1087e-01, 2.0473e01, 2.0104e01],
        [6.2832e-01, 2.0865e01, 2.0501e01],
        [6.4577e-01, 2.1257e01, 2.0899e01],
        [6.6323e-01, 2.1650e01, 2.1296e01],
        [6.8068e-01, 2.2042e01, 2.1693e01],
        [6.9813e-01, 2.2434e01, 2.2091e01],
        [7.1558e-01, 2.2826e01, 2.2488e01],
        [7.3304e-01, 2.3218e01, 2.2886e01],
        [7.5049e-01, 2.3610e01, 2.3283e01],
        [7.6794e-01, 2.4002e01, 2.3680e01],
        [7.8540e-01, 2.4395e01, 2.4078e01],
        [8.0285e-01, 2.4787e01, 2.4475e01],
        [8.2030e-01, 2.5179e01, 2.4872e01],
        [8.3776e-01, 2.5571e01, 2.5270e01],
        [8.5521e-01, 2.5963e01, 2.5667e01],
        [8.7266e-01, 2.6355e01, 2.6065e01],
        [8.9012e-01, 2.6747e01, 2.6462e01],
        [9.0757e-01, 2.7140e01, 2.6859e01],
        [9.2502e-01, 2.7532e01, 2.7257e01],
        [9.4248e-01, 2.7924e01, 2.7654e01],
        [9.5993e-01, 2.8316e01, 2.8051e01],
        [9.7738e-01, 2.8708e01, 2.8449e01],
        [9.9484e-01, 2.9100e01, 2.8846e01],
        [1.0123e00, 2.9492e01, 2.9244e01],
        [1.0297e00, 2.9885e01, 2.9641e01],
        [1.0472e00, 3.0277e01, 3.0038e01],
        [1.0647e00, 3.0669e01, 3.0436e01],
        [1.0821e00, 3.1061e01, 3.0833e01],
        [1.0996e00, 3.1453e01, 3.1230e01],
        [1.1170e00, 3.1845e01, 3.1628e01],
        [1.1345e00, 3.2237e01, 3.2025e01],
        [1.1519e00, 3.2630e01, 3.2423e01],
        [1.1694e00, 3.3022e01, 3.2820e01],
        [1.1868e00, 3.3414e01, 3.3217e01],
        [1.2043e00, 3.3806e01, 3.3615e01],
        [1.2217e00, 3.4198e01, 3.4012e01],
        [1.2392e00, 3.4590e01, 3.4409e01],
        [1.2566e00, 3.4982e01, 3.4807e01],
        [1.2741e00, 3.5375e01, 3.5204e01],
        [1.2915e00, 3.5767e01, 3.5602e01],
        [1.3090e00, 3.6159e01, 3.5999e01],
        [1.3265e00, 3.6551e01, 3.6396e01],
        [1.3439e00, 3.6943e01, 3.6794e01],
        [1.3614e00, 3.7335e01, 3.7191e01],
        [1.3788e00, 3.7727e01, 3.7589e01],
        [1.3963e00, 3.8120e01, 3.7986e01],
        [1.4137e00, 3.8512e01, 3.8383e01],
        [1.4312e00, 3.8904e01, 3.8781e01],
        [1.4486e00, 3.9296e01, 3.9178e01],
        [1.4661e00, 3.9688e01, 3.9575e01],
        [1.4835e00, 4.0080e01, 3.9973e01],
        [1.5010e00, 4.0472e01, 4.0370e01],
        [1.5184e00, 4.0865e01, 4.0768e01],
        [1.5359e00, 4.1257e01, 4.1165e01],
        [1.5533e00, 4.1649e01, 4.1562e01],
        [1.5708e00, 4.2041e01, 4.1960e01],
         [1.5808e00, 4.2441e01, 4.2360e01],
    ],
    dtype=torch.float64,
)

class MaxLoss(nn.Module):
    def __init__(self, N=100, lookup_table=None):
        super(MaxLoss, self).__init__()
        self.N = N
        self.lookup_table = lookup_table[:, :2]
        self.lookup_table_x = lookup_table[:, 0]
        self.lookup_table_y = lookup_table[:, 1]

    def forward(self, y_pred, y_true):
        diff = torch.abs(y_pred - y_true)
        diff = torch.norm(diff, dim=1)
        C = torch.mean(diff)
        scaled_diff = torch.clamp(self.N * (diff - C), min=-100, max=100)

        max_approx = (1 / self.N) * torch.log(torch.exp(scaled_diff).sum(dim=0) + 1e-8) + C 

        lower_indices = (
            torch.searchsorted(self.lookup_table_x, max_approx, right=False) - 1
        )
        upper_indices = lower_indices + 1
        upper_indices = torch.clamp(upper_indices, max=self.lookup_table_x.size(0) - 1)
        if lower_indices == self.lookup_table_x.size(0)-1:
            upper_indices = lower_indices
            lower_indices-=1

        x0 = self.lookup_table_x[lower_indices]
        x1 = self.lookup_table_x[upper_indices]
        y0 = self.lookup_table_y[lower_indices]
        y1 = self.lookup_table_y[upper_indices]
        
        weight = (max_approx - x0) / (x1 - x0)
        latency = (y0 + weight * (y1 - y0))
        return latency

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 


def parse_args():
    parser = argparse.ArgumentParser(description="Train a gaze estimation model.")
    parser.add_argument(
        "--batch_size", type=int, default=800, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.000005, help="learning rate")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--val_folder",
        type=str,
        default="../openeds/validation/sequences1/",
        help="training folder",
    )
    parser.add_argument(
        "--train_folder",
        type=str,
        default="../openeds/train/sequences1/",
        help="training folder",
    )
    parser.add_argument(
        "--info_file",
        type=str,
        default="./data/validation/validation/val.csv",
        help="Information File",
    )
    parser.add_argument("--output_dir", type=str, default="results", help="output path")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="Directory to save tensorboard logs",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Proportion of data to use for validation",
    )
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random Seed")
    parser.add_argument("--inference", type=bool, default=False, help="Random Seed")

    return parser.parse_args()


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def split_folders(image_folders, train_ratio=0.9):
    train_folders, val_folders = train_test_split(
        image_folders, train_size=train_ratio, random_state=42
    )
    return train_folders, val_folders

def gather_image_folders(main_folder):
    return [os.path.join(main_folder, d) for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]

def split_folders(image_folders, train_ratio=0.8):
    return train_test_split(image_folders, train_size=train_ratio, random_state=42)


def main():
    args = parse_args()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, "model_minmax_4_updated.pt")

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, current_time)
    trian_folders = gather_image_folders(args.train_folder)
    val_folders = gather_image_folders(args.val_folder)

    train_dataset = AllEdsDataset(image_folder=train_folders)
    val_dataset = AllEdsDataset(image_folder=val_folders)

    dataloaders = {
        "train": DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        ),
        "val": DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        ),
    }

    model = VisionTransformer(num_layers = 4, top_k = 1).to(device)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_minmax_4.pt")))

    if not args.inference:

        criterion = nn.L1Loss()

        criterion_train = MaxLoss(N=100, lookup_table = lookup_table.to(device))
        # criterion_train = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

        train_model(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion_train,
            criterion_val=criterion,
            optimizer=optimizer,
            device=device,
            log_dir=args.log_dir,
            output_path=output_path,
            num_epochs=args.num_epochs,
            scheduler=scheduler,
        )
    else:
        with torch.no_grad():
            for images, _, _, gaze_gt_vecs in dataloaders["val"]:
                images = images.to(device)
                gaze_gt_vecs = gaze_gt_vecs.to(device)
                output = model(images)
                print(gaze_gt_vecs[:5])
                print(output[:5])


if __name__ == "__main__":
    main()
