import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class QNetwork(nn.Module):
    def __init__(self, actions):
        super(QNetwork, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=False)
        self.base_model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # freeze layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(self.base_model.classifier[1].in_features, actions)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
