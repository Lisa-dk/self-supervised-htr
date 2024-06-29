import torch
from torch import nn
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, model_name, edisdist):
        super(SiameseNetwork, self).__init__()
        self.mode = edisdist

        if model_name == "vgg16":
            self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.model.classifier = self.model.classifier[:] # remove output layer
        if model_name == "vgg11":
            self.model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
            self.model.classifier = self.model.classifier[:] # remove output layer
        if model_name == "resnet34":
            self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            
            if self.mode:
                self.model.fc = nn.Sequential(nn.Identity())
                self.fc = nn.Sequential(nn.Linear(in_features=1024, out_features=512), nn.ReLU(), nn.BatchNorm1d(256),
                                        nn.Linear(in_features=512, out_features=1))
            else:
                self.model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=256), nn.ReLU(), nn.BatchNorm1d(256),
                                        nn.Linear(in_features=256, out_features=256))
        if model_name == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Sequential(nn.Linear(in_features=512, out_features=256), nn.ReLU(), 
                                        nn.Linear(in_features=256, out_features=256))
    
    def forward_pass(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x)
        x = self.model.classifier(x)
        return x
    
    def forward(self, x1, x2):
        x1 = self.model(x1)
        x2 = self.model(x2)

        if self.mode:
            x = torch.cat((x1, x2), dim=1)
            x = self.fc(x)
            return x
        else:
            return x1, x2