import torch
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from matplotlib import pyplot
import numpy as np


def generate_sampler(dataset, indices, num_of_labels):
    train_labels = []  # list of labels of any example in train set
    class_counts = [0] * num_of_labels
    num_samples = len(indices)
    for idx in indices:
        _, label = dataset[idx]
        class_counts[label] += 1
        train_labels.append(label)

    class_weights = [num_samples / class_count for class_count in class_counts]
    # give a weight for any example in the train set
    weights = [class_weights[train_labels[i]] for i in range(num_samples)]
    sampler = torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(weights), num_samples=num_samples)
    return sampler


def train_epoch(model, device, dataloader, loss_fn, optimizer):

    model.train()
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        output = model(images)

        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_model(model, device, dataloader):
    total = 0
    accuracy = 0.0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

            tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
            true_positive += tp
            true_negative += tn
            false_positive += fp
            false_negative += fn

    accuracy = (100 * accuracy / total)
    sensitivity = true_positive/(true_positive + false_negative)
    specificity = true_negative/(true_negative + false_positive)
    return accuracy, sensitivity, specificity


def dataloader_roc(model, device, dataloader, titel):
    model.eval()
    y_true = []
    y_score = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            y_true = np.append(y_true, labels.detach().numpy())
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)[:, 1]
            y_score = np.append(y_score, probabilities.detach().numpy())

    fpr, tpr, _ = roc_curve(y_true, y_score)
    pyplot.plot(fpr, tpr, marker='.')

    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

    pyplot.savefig(f'roc_curves/{titel}.jpg')
    pyplot.show()
    pyplot.close()



