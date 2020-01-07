import torch
import torch.nn.functional as F


class EasyModel(torch.nn.Module):
    def __init__(self):
        super(EasyModel, self).__init__()
        self.fc = torch.nn.Linear(28*28, 10)

    def forward(self, x):
        x = self.fc(x)
        return x


class MediumModel(torch.nn.Module):
    def __init__(self):
        super(MediumModel, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 200)
        self.fc2 = torch.nn.Linear(200, 200)
        self.fc3 = torch.nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)


# This network topology was inspired by https://github.com/meghanabhange/FashionMNIST-3-Layer-CNN/issues/new
class AdvancedModel(torch.nn.Module):
    def __init__(self):
        super(AdvancedModel, self).__init__()

        self.cnn1 = torch.nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = torch.nn.ReLU()
        self.conv1_bn = torch.nn.BatchNorm2d(16)
        self.MaxPool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.cnn2 = torch.nn.Conv2d(in_channels=16, out_channels=32,
                                    kernel_size=5, stride=1, padding=2)
        self.relu2 = torch.nn.ReLU()
        self.conv2_bn = torch.nn.BatchNorm2d(32)

        self.MaxPool2 = torch.nn.MaxPool2d(kernel_size=2)

        self.cnn3 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                                    kernel_size=5, stride=1, padding=2)
        self.relu3 = torch.nn.ReLU()
        self.conv3_bn = torch.nn.BatchNorm2d(64)

        self.MaxPool3 = torch.nn.MaxPool2d(kernel_size=2)

        self.fc1 = torch.nn.Linear(576, 10)

    def forward(self, x):

        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.conv1_bn(out)

        out = self.MaxPool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.conv2_bn(out)

        out = self.MaxPool2(out)

        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.conv3_bn(out)

        out = self.MaxPool3(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out
