import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name, num_classes=10):
    if model_name == 'simcnn':
        from models.simcnn import SimCNN
        model = SimCNN(num_classes=num_classes)
    elif model_name == 'resnet18':
        from models.resnet18 import ResNet18
        model = ResNet18(num_classes=num_classes)
    elif model_name == 'mobilenet':
        from models.mobilenet import MobileNetV1
        model = MobileNetV1(num_classes=num_classes)
    else:
        raise ValueError
    return model


def load_trained_model(model_name, n_classes, trained_model_path):
    model = load_model(model_name, num_classes=n_classes)
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model
