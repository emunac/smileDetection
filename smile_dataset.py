from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image


class SmileDataset(Dataset):
    def __init__(self, img_dir, positive_path, negative_path, transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        self.img_labels = []
        self.transform = transform

        for path in [positive_path, negative_path]:
            for file in os.listdir(path):
                if file.endswith(".jpg"):
                    image_path = os.path.join(path, file)
                    self.img_paths.append(image_path)
                    if path is positive_path:
                        self.img_labels.append(1)
                    else:
                        self.img_labels.append(0)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        convert_tensor = transforms.ToTensor()
        image = Image.open(self.img_paths[idx])
        image = convert_tensor(image)
        if self.transform:
            image = self.transform(image)

        return image, self.img_labels[idx]
