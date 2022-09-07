import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import os
from torchvision import transforms
from torch import nn
import torch.optim as optim
from smile_net import Net
from smile_dataset import SmileDataset
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import model_utils

torch.manual_seed(2)

PATH = 'state_dict_models/'
img_dir = '../SMILEsmileD/SMILEs'
positive_path = os.path.join(img_dir, 'positives/positives7')
negative_path = os.path.join(img_dir, 'negatives/negatives7')

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2)
])

dataset = SmileDataset(img_dir, positive_path, negative_path, transform=transform)
print(f'Numbers of images in dataset: {len(dataset)}')


parameters = dict(
    lr=[0.001, 0.003, 0.0001],
    train_batch_size=[32, 64, 128]
)
param_values = [v for v in parameters.values()]

# Configuration options
loss_function = nn.NLLLoss()
n_epochs = 50

train_size = int(0.8 * len(dataset))
train_batch_size = 128
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

sampler = model_utils.generate_sampler(dataset, train_dataset.indices, 2)


for ixd, (lr, train_batch_size) in enumerate(product(*param_values)):

    print(f'start training for hyperparameters: lr = {lr}, batch size = {train_batch_size}')
    print(f'run num: {ixd + 1}')

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, sampler=sampler)
    val_loader = DataLoader(dataset=val_dataset, batch_size=20)

    comment = f' batch_size = {train_batch_size} lr = {lr}'
    tb = SummaryWriter(comment=comment)

    max_accuracy = 0
    max_sensitivity = 0
    epochs_max_sens = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(n_epochs):

        model_utils.train_epoch(model, device, train_loader, loss_function, optimizer)
        accuracy, sensitivity, specificity = model_utils.test_model(model, device, val_loader)
        tb.add_scalar('Accuracy', accuracy, epoch)
        tb.add_scalar('Sensitivity', sensitivity, epoch)
        tb.add_scalar('Specificity', specificity, epoch)

        if sensitivity > max_sensitivity:
            max_sensitivity = sensitivity
            epochs_max_sens.append(epoch)
            torch.save(model.state_dict(), f'{PATH}sensitivity_lr={lr}_b_size={train_batch_size}.pt')

        if accuracy > max_accuracy:
            max_accuracy = accuracy

        print(
            f' epoch {epoch + 1}, accuracy={accuracy:.2f}, sensitivity={sensitivity:.2}, specificity={specificity:.2f}')

    print(f'sensitivity get max at steps: {epochs_max_sens}')
    # plot roc curve at max sensitivity model
    print(f'roc_curve for lr={lr}, b_size={train_batch_size}:')
    model.load_state_dict(torch.load(f'{PATH}sensitivity_lr={lr}_b_size={train_batch_size}.pt'))
    model_utils.dataloader_roc(model, device, val_loader, titel=f'lr={lr}_b_size={train_batch_size}')

    tb.add_hparams(
        {'lr': lr, 'bsize': train_batch_size},
        {
            'accuracy': max_accuracy,
            'sensitivity': max_sensitivity,
        },
    )
tb.close()

