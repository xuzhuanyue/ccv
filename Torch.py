import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

net = models.vgg16()


class VGGnet(nn.Module):
    def __init__(self, feature_extract=True, num_classes=5):
        super(VGGnet, self).__init__()
        # 导入VGG16模型
        model = models.vgg16(pretrained=True)
        # 加载features部分
        self.features = model.features
        # 固定特征提取层参数
        set_parameter_requires_grad(self.features, feature_extract)
        # 加载avgpool层
        self.avgpool = model.avgpool
        # 改变classifier：分类输出层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        out = self.classifier(x)
        return out


# 固定参数，不进行训练
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
