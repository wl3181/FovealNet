import torch
from torch.utils.data import DataLoader, Dataset, Subset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def convert_gaze(vector_str) -> np.ndarray:
    try:
        parts = vector_str.split("_")
        if len(parts) != 3:
            raise ValueError(f"Invalid vector length: {len(parts)}")
        x, y, z = map(float, parts)
        pitch = np.arcsin(-y)
        yaw = np.arctan2(x, z)
        return np.array([pitch, yaw]).astype(np.float32)
    except Exception as e:
        raise ValueError(f"Error converting gaze vector '{vector_str}': {e}")


class EdsDataset(Dataset):
    def __init__(
        self, image_folder, info_file, transform_left=None, transform_right=None
    ):
        self.image_folder = image_folder
        self.info_file = pd.read_csv(info_file)

        if transform_left is None or transform_right is None:
            transform_left = transforms.Compose(
                [
                    # transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )

            transform_right = transforms.Compose(
                [
                    # transforms.Resize((224, 224)),
                    transforms.ColorJitter(brightness=0.5),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                ]
            )
        self.transform_left = transform_left
        self.transform_right = transform_right

    def __len__(self):
        return len(self.info_file)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.image_folder, "sequences", self.info_file.iloc[idx]["image"] + ".png"
        )
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        eye_type = self.info_file.iloc[idx]["eye_type"]

        if eye_type == "left":
            image = self.transform_left(image)
        else:
            image = self.transform_right(image)

        gaze_gt_vec = self.info_file.iloc[idx]["gaze_gt_vec"]
        gaze_gt_vec = convert_gaze(gaze_gt_vec)

        rec_type = self.info_file.iloc[idx]["rec_type"]

        return image, gaze_gt_vec, eye_type, rec_type

class AllEdsDataset(Dataset):

    def __init__(
        self, image_folder, transform_left=None, transform_right=None, split="train"
    ):
        self.image_folder = image_folder
        self.split = split

        if transform_left is None or transform_right is None:
            transform_left = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                    transforms.ColorJitter(brightness=0.5),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.225]),
                ]
            )

            transform_right = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                    transforms.ColorJitter(brightness=0.5),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.225]),
                ]
            )
        self.transform_left = transform_left
        self.transform_right = transform_right

        self.data = []
        for folder in image_folder:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith(".png"):
                        file_base = os.path.splitext(file)[0]
                        parts = file_base.split("_")
                        if len(parts) >= 6:
                            try:
                                rec_type = parts[1]
                                eye_type = parts[2]
                                gaze_direction_str = "_".join(parts[3:6])
                                gaze_direction = convert_gaze(gaze_direction_str)
                                full_path = os.path.join(root, file)
                                self.data.append(
                                    {
                                        "image_path": full_path,
                                        "rec_type": rec_type,
                                        "eye_type": eye_type,
                                        "gaze_direction": gaze_direction,
                                    }
                                )
                            except ValueError:
                                print(f"Skipping file due to parsing error: {file}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image_path"]
        rec_type = sample["rec_type"]
        eye_type = sample["eye_type"]
        gaze_direction = sample["gaze_direction"]

        image = Image.open(image_path).convert("L")
        if eye_type == "left":
            image = self.transform_left(image)
        else:
            image = self.transform_right(image)

        return image, rec_type, eye_type, torch.tensor(gaze_direction)



def train_model(
    model,
    dataloaders,
    criterion,
    criterion_val,
    optimizer,
    device,
    log_dir,
    output_path,
    scheduler,
    num_epochs=50,
    batch_size=256,
    patience=50,  # Number of epochs to wait for improvement
):
    print(device)
    model.to(device)
    writer = SummaryWriter(log_dir=log_dir)
    best_val_loss = float("inf")
    epochs_no_improve = 0  # Counter for epochs without improvement
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered")
            break

        model.train()
        running_loss = 0.0
        max_error = 0.0
        vertical_loss = 0.0
        horizon_loss = 0.0
        min_error = float("inf")

        progress_bar = tqdm(dataloaders["train"], desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (images, _, _, gaze_gt_vecs) in enumerate(progress_bar):
            images = images.to(device)
            gaze_gt_vecs = gaze_gt_vecs.to(device)

            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output, gaze_gt_vecs)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            abs_error = torch.abs(output - gaze_gt_vecs)
            batch_max_error = torch.max(abs_error).item()
            batch_min_error = torch.min(abs_error).item()

            max_error = max(max_error, batch_max_error)
            min_error = min(min_error, batch_min_error)
            vertical_loss += criterion_val(output[:, 0], gaze_gt_vecs[:, 0]).item()
            horizon_loss += criterion_val(output[:, 1], gaze_gt_vecs[:, 1]).item()

            progress_bar.set_postfix(loss=running_loss / (i + 1))

            global_step = epoch * len(dataloaders["train"]) + i
            writer.add_scalar("Loss/Train", loss.item(), global_step)
        scheduler.step()

        epoch_loss = running_loss / len(dataloaders["train"])
        vertical_loss /= len(dataloaders["train"])
        horizon_loss /= len(dataloaders["train"])


        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Max Error: {max_error:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Min Error: {min_error:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Vertical Loss: {vertical_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Horizon Loss: {horizon_loss:.4f}")

        writer.add_scalar("Loss/Epoch", epoch_loss, epoch)
        writer.add_scalar("Error/Max", max_error, epoch)
        writer.add_scalar("Error/Min", min_error, epoch)

        model.eval()
        val_loss = 0.0
        val_max_error = 0.0
        val_min_error = float("inf")
        vertical_loss = 0.0
        horizon_loss = 0.0
        with torch.no_grad():
            for images, _, _, gaze_gt_vecs in dataloaders["val"]:
                images = images.to(device)
                gaze_gt_vecs = gaze_gt_vecs.to(device)

                output = model(images)
                loss = criterion(output, gaze_gt_vecs)

                val_loss += loss.item()

                abs_error = torch.abs(output - gaze_gt_vecs)
                abs_error = torch.norm(abs_error, dim=1)
                batch_max_error = torch.max(abs_error).item()
                batch_min_error = torch.min(abs_error).item()

                val_max_error = max(val_max_error, batch_max_error)
                val_min_error = min(val_min_error, batch_min_error)

                vertical_loss += criterion_val(output[:, 0], gaze_gt_vecs[:, 0]).item()
                horizon_loss += criterion_val(output[:, 1], gaze_gt_vecs[:, 1]).item()

        val_loss /= len(dataloaders["val"])
        vertical_loss /= len(dataloaders["val"])
        horizon_loss /= len(dataloaders["val"])

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Vertical Loss: {vertical_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Horizon Loss: {horizon_loss:.4f}")
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Validation Max Error: {val_max_error:.4f}"
        )
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Validation Min Error: {val_min_error:.4f}"
        )

        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Error/Validation_Max", val_max_error, epoch)
        writer.add_scalar("Error/Validation_Min", val_min_error, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  # Reset counter if validation loss improves
            torch.save(model.state_dict(), output_path)
            print(f"Model saved to {output_path}")
        else:
            epochs_no_improve += 1
            print(f"Epochs without improvement: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            early_stop = True
            print("Early stopping triggered due to no improvement in validation loss")

    print("Finished Training")
    writer.close()



if __name__ == "__main__":
    image_folder = "/root/autodl-tmp/validation_1"
    dataset = AllEdsDataset(image_folder)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for images, rec_type, eye_type, gaze_directions in dataloader:
        print(
            f"Images shape: {images.shape}, Gaze Directions: {gaze_directions}, Rec Type: {rec_type}, Eye Type: {eye_type}"
        )
        break 
