import torch
from tqdm import tqdm
from prettytable import PrettyTable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, test_loader, acc_in_percent=False):
    model.to(DEVICE)
    model.eval()
    total_correct = 0
    total_samples = 0
    num_classes = len(test_loader.dataset.classes)

    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.no_grad():
        for batch_inputs, batch_labels in tqdm(test_loader, desc="Evaluating"):
            batch_inputs, batch_labels = batch_inputs.to(DEVICE), batch_labels.to(DEVICE)
            outputs = model(batch_inputs)
            pred = torch.argmax(outputs, dim=1)
            for i in range(batch_labels.size(0)):
                label = batch_labels[i].item()
                class_total[label] += 1
                if pred[i].item() == label:
                    class_correct[label] += 1

    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    per_class_accuracy = [
        class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(num_classes)
    ]
    if acc_in_percent:
        accuracy *= 100
        per_class_accuracy = [x * 100 for x in per_class_accuracy]
    return accuracy, per_class_accuracy


@torch.no_grad()
def evaluate_model_for_target_classes(model, data_loader, target_classes, acc_in_percent=False):
    model.eval()
    n_correct, total_labels = 0, 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        if target_classes is not None:
            outputs = outputs[:, target_classes]

        predicts = torch.argmax(outputs, dim=1)
        n_correct += torch.sum((predicts == labels).float())
        total_labels += len(labels)
    accuracy = n_correct / total_labels
    if acc_in_percent:
        accuracy *=100
    return accuracy

def print_model_summary(model):
    columns = ["Modules", "Parameters", "Param Shape"]
    table = PrettyTable(columns)
    for i, col in enumerate(columns):
        if i == 0:
            table.align[col] = "l"
        else:
            table.align[col] = "r"
    total_param_nums = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_nums = parameter.numel()
        param_shape = list(parameter.shape)
        table.add_row([name, "{:,}".format(param_nums), "{}".format(param_shape)])
        total_param_nums += param_nums

    separator = ["-" * len(x) for x in table.field_names]
    table.add_row(separator)
    table.add_row(["Total", "{:,}".format(total_param_nums), "{}".format("_")])

    print(table, "\n")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item