import torch
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import confusion_matrix


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


def test_model(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total = 0
    accuracy = 0.0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
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
