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

torch.manual_seed(2)

# def show(imgs):
#   grid = make_grid(imgs, nrow=5)
#   plt.imshow(grid.permute(1, 2, 0))
#   plt.show()

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(3380, 50)
    self.fc2 = nn.Linear(50, 2)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 3380)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


class FaceDataset(Dataset):
  def __init__(self, img_dir):
    self.img_dir = img_dir
    self.img_paths = []
    self.img_labels = []

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
    image_tensor = convert_tensor(image)
    return image_tensor, self.img_labels[idx]


def test_validation():

  model.eval()
  total = 0
  accuracy = 0.0
  true_positive = 0
  false_negative = 0

  with torch.no_grad():
    for images, labels in val_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      accuracy += (predicted == labels).sum().item()

      tn, fp, fn, tp = confusion_matrix(labels, predicted).ravel()
      true_positive += tp
      false_negative += fn

  accuracy = (100 * accuracy / total)
  sensitivity = true_positive/(true_positive + false_negative)
  return accuracy, sensitivity


img_dir = "../SMILEsmileD/SMILEs"
dataset = FaceDataset(img_dir)
print("Numbers of images in dataset:", len(dataset))
print("One example from dataset:", dataset[0])
plt.imshow(dataset[0][0].permute(1, 2, 0))
plt.show()

train_size = int(0.8 * len(dataset))
train_batch_size = 128
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#generate sampler
train_labels = [] #list of labels of any axample in train set
class_counts = [0] * 2
for idx in train_dataset.indices:
  _, label = dataset[idx]
  class_counts[label] += 1
  train_labels.append(label)

class_weights = [train_size/class_counts[i] for i in range(len(class_counts))]
#give a weight for any example in the train set
weights = [class_weights[train_labels[i]] for i in range(int(train_size))]
sampler = torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(weights), train_size)

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, sampler=sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)

n_epochs = 1000
lr = 0.001
model = Net()
optimizer = optim.Adam(model.parameters(), lr=lr)

max_accuracy = 0
PATH = "state_dict_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

  accuracy, sensitivity = test_validation()
  if accuracy > max_accuracy:
    max_accuracy = accuracy
    torch.save(model.state_dict(), PATH)

  print(epoch + 1, "lr =", lr, "loss_train:", torch.stack(losses).mean(),
        "accuracy:", accuracy, "sensitivity:", sensitivity)

