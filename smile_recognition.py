import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import os
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from smile_net import Net
from smile_dataset import SmileDataset
from torch.utils.tensorboard import SummaryWriter
from itertools import product
from sklearn.model_selection import KFold
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


# Later
# parameters = dict(
#     lr=[0.1, 0.01, 0.001, 0.0001],
#     train_batch_size=[32, 64, 128]
# )
#
# param_values = [v for v in parameters.values()]
# print(param_values)
#
# for lr, train_batch_size in product(*param_values):
#     print(lr, train_batch_size)

# Configuration options
loss_function = nn.NLLLoss()
k_folds = 5
n_epochs = 50
lr = 0.001
train_batch_size = 128

# For fold results
results = {}

kfold = KFold(n_splits=k_folds, shuffle=True)

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

    print(f'fold id: {fold + 1}')
    train_dataset = Subset(dataset=dataset, indices=train_ids)
    test_dataset = Subset(dataset=dataset, indices=test_ids)

    train_sampler = model_utils.generate_sampler(dataset, train_dataset.indices, 2)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=20)

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    max_accuracy = 0
    max_sensitivity = 0
    PATH = "state_dict_model_sensitivity.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(n_epochs):
        model.train()
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)

            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        accuracy, sensitivity, specificity = model_utils.test_model(model, test_loader)

        if sensitivity > max_sensitivity:
            max_sensitivity = sensitivity
            # torch.save(model.state_dict(), PATH)

        if accuracy > max_accuracy:
            max_accuracy = accuracy

        print(
            f' epoch {epoch + 1}, accuracy={accuracy:.2f}, sensitivity={sensitivity:.2}, specificity={specificity:.2f}')

    results[fold] = max_accuracy

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')

sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
print(f'Average: {sum/len(results.items())} %')

