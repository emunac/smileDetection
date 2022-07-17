import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import os
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from smile_net import Net
from smile_dataset import SmileDataset
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import model_utils

torch.manual_seed(2)

img_dir = "../SMILEsmileD/SMILEs"
positive_path = os.path.join(img_dir, "positives/positives7")
negative_path = os.path.join(img_dir, "negatives/negatives7")

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2)
])

dataset = SmileDataset(img_dir, positive_path, negative_path, transform=transform)
print("Numbers of images in dataset:", len(dataset))
print("One example from dataset:", dataset[0])
plt.imshow(dataset[0][0].permute(1, 2, 0))
plt.show()

parameters = dict(
    lr=[0.1, 0.01, 0.001, 0.0001],
    train_batch_size=[32, 64, 128]
)

param_values = [v for v in parameters.values()]
print(param_values)

for lr, train_batch_size in product(*param_values):
    print(lr, train_batch_size)

train_size = int(0.8 * len(dataset))
train_batch_size = 128
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

sampler = model_utils.generate_sampler(dataset, train_dataset.indices, 2)

train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, sampler=sampler)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)

tb = SummaryWriter()


n_epochs = 50
lr = 0.001
model = Net()
optimizer = optim.Adam(model.parameters(), lr=lr)

images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)
tb.add_image("images", grid)
tb.add_graph(model, images)
tb.close()

max_accuracy = 0
max_sensitivity = 0
PATH = "state_dict_model_sensitivity.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(n_epochs):
    model.train()
    losses = []
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        output = model(images)

        loss = F.nll_loss(output, labels)
        losses.append(loss.detach())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    accuracy, sensitivity, specificity = model_utils.test_model(model, val_loader)
    tb.add_scalar('Accuracy', accuracy, epoch)
    tb.add_scalar('Sensitivity', sensitivity, epoch)
    tb.add_scalar('Specificity', specificity, epoch)

    if sensitivity > max_sensitivity:
        max_sensitivity = sensitivity
        # torch.save(model.state_dict(), PATH)

    print(f' epoch {epoch + 1}, accuracy={accuracy:.2f}, sensitivity={sensitivity:.2}, specificity={specificity:.2f}')
