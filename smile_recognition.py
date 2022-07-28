import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import os
from torchvision import transforms
from torch import nn
import torch.optim as optim
from smile_net import Net
from smile_dataset import SmileDataset
from torch.utils.tensorboard import SummaryWriter
from itertools import product
from sklearn.model_selection import KFold
import model_utils

torch.manual_seed(2)


img_dir = '../SMILEsmileD/SMILEs'
positive_path = os.path.join(img_dir, 'positives/positives7')
negative_path = os.path.join(img_dir, 'negatives/negatives7')

transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2)
])

dataset = SmileDataset(img_dir, positive_path, negative_path, transform=transform)
print(f'Numbers of images in dataset: {len(dataset)}')

tb = SummaryWriter()

parameters = dict(
    lr=[0.001, 0.003, 0.0001],
    train_batch_size=[32, 64, 128]
)
param_values = [v for v in parameters.values()]

# Configuration options
loss_function = nn.NLLLoss()
k_folds = 5
n_epochs = 50

# For fold results
results = {}
id_run = 0

for ixd, (lr, train_batch_size) in enumerate(product(*param_values)):

    print(f'start training for hyperparameters: lr = {lr}, batch size = {train_batch_size}')
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        print(f'run num: {id_run + 1} fold id: {fold + 1}')

        train_dataset = Subset(dataset=dataset, indices=train_ids)
        test_dataset = Subset(dataset=dataset, indices=test_ids)

        train_sampler = model_utils.generate_sampler(dataset, train_dataset.indices, 2)
        train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset=test_dataset, batch_size=20)

        model = Net()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        max_accuracy = 0
        max_sensitivity = 0
        PATH = 'state_dict_model_sensitivity.pt'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        for epoch in range(n_epochs):

            model_utils.train_epoch(model, device, train_loader, loss_function, optimizer)
            accuracy, sensitivity, specificity = model_utils.test_model(model, device, test_loader)
            tb.add_scalar('accuracy', accuracy, epoch)
            tb.add_scalar('sensitivity', sensitivity, epoch)
            tb.add_scalar('specificity', specificity, epoch)

            if sensitivity > max_sensitivity:
                max_sensitivity = sensitivity
                # torch.save(model.state_dict(), PATH)

            if accuracy > max_accuracy:
                max_accuracy = accuracy

            print(
                f' epoch {epoch + 1}, accuracy={accuracy:.2f}, sensitivity={sensitivity:.2}, specificity={specificity:.2f}')

        results[fold] = max_accuracy
        id_run += 1

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')

    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')

    tb.add_hparams(
        {'lr': lr, 'bsize': train_batch_size},
        {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
        },
    )
tb.close()

