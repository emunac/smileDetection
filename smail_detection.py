import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from torchvision import transforms
from PIL import Image


class FaceDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_paths = []
        self.img_labels = []

        #init positives examples
        positive_path = os.path.join(img_dir, "positives/positives7")
        for file in os.listdir(positive_path):
            if file.endswith(".jpg"):
                image_path = os.path.join(positive_path, file)
                self.img_paths.append(image_path)
                self.img_labels.append(1)

        #init negatives examples
        negative_path = os.path.join(img_dir, "negatives/negatives7")
        for file in os.listdir(negative_path):
            if file.endswith(".jpg"):
                image_path = os.path.join(negative_path, file)
                self.img_paths.append(image_path)
                self.img_labels.append(0)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        convert_tensor = transforms.ToTensor()
        image = Image.open(self.img_paths[idx])
        image_tensor = convert_tensor(image)
        return image_tensor, self.img_labels[idx]


img_dir = "../SMILEsmileD/SMILEs"
dataset = FaceDataset(img_dir)
print("Numbers of images in dataset:", len(dataset))
print("One example from dataset:", dataset[0])

