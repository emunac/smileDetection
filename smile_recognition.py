import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision.utils import make_grid
import os
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from smile_net import Net

torch.manual_seed(2)

# def show(imgs):
#   grid = make_grid(imgs, nrow=5)
#   plt.imshow(grid.permute(1, 2, 0))
#   plt.show()


class FaceDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        self.img_labels = []
        self.transform = transform

        positive_path = os.path.join(img_dir, "positives/positives7")
        negative_path = os.path.join(img_dir, "negatives/negatives7")
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


def generate_sampler(dataset, indices):
    train_labels = []  # list of labels of any example in train set
    class_counts = [0] * 2
    num_samples = len(indices)
    for idx in indices:
        _, label = dataset[idx]
        class_counts[label] += 1
        train_labels.append(label)

    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
    # give a weight for any example in the train set
    weights = [class_weights[train_labels[i]] for i in range(num_samples)]
    sampler = torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=num_samples)
    return sampler


def wrap_confusion_matrix(y_true, y_pred):
    matrix = []     # tn, fp, fn, tp
    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()

    # expected
    if len(matrix) == 4:
        return matrix

    # else len = 1 which means that y_true = y_pred
    elif y_true[0] == 0:
        return [4, 0, 0, 0]
    else:
        return [0, 0, 0, 4]


def test_validation():
    model.eval()
    total = 0
    accuracy = 0.0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

            tn, fp, fn, tp = wrap_confusion_matrix(labels, predicted)
            true_positive += tp
            true_negative += tn
            false_positive += fp
            false_negative += fn

    accuracy = (100 * accuracy / total)
    sensitivity = true_positive/(true_positive + false_negative)
    specificity = true_negative/(true_negative + false_positive)
    return accuracy, sensitivity, specificity


img_dir = "../SMILEsmileD/SMILEs"

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2)
])

dataset = FaceDataset(img_dir, transform=transform)
print("Numbers of images in dataset:", len(dataset))
print("One example from dataset:", dataset[0])
plt.imshow(dataset[0][0].permute(1, 2, 0))
plt.show()

train_size = int(0.8 * len(dataset))
train_batch_size = 128
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


sampler = generate_sampler(dataset, train_dataset.indices)

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, sampler=sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)

n_epochs = 1000
lr = 0.001
model = Net()
optimizer = optim.Adam(model.parameters(), lr=lr)

max_accuracy = 0
max_sensitivity = 0
PATH = "state_dict_model_sensitivity.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.load_state_dict(torch.load(PATH, map_location=device))
test_validation()

for epoch in range(n_epochs):
    model.train()
    losses = []
    for image, label in tqdm(train_loader):
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        loss = F.nll_loss(output, label)
        losses.append(loss.detach())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    accuracy, sensitivity, specificity = test_validation()
    if sensitivity > max_sensitivity:
        max_sensitivity = sensitivity
        torch.save(model.state_dict(), PATH)

    print(epoch + 1, "lr =", lr, "loss_train:", torch.stack(losses).mean(),
          "accuracy:", accuracy, "sensitivity:", sensitivity, "specificity:", specificity)

