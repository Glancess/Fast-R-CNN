import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16
from utils.RoIPool import RoIPool


class FastRCNN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 用 ImageNet 预训练的 VGG16 初始化特征提取部分。
        # 加载自己训练好的 checkpoint 时，不必再次下载预训练权重。
        weights = VGG16_Weights.DEFAULT if pretrained else None
        vgg = vgg16(weights=weights)

        self.backbone = vgg.features[:-1]
        self.classifier = vgg.classifier[:-1]

        self.roi_pool = RoIPool()

        self.clshead = nn.Linear(4096, 21)
        self.bboxhead = nn.Linear(4096, 4 * 20)
        self.flatten = nn.Flatten()

        # 新增的两个输出层没有预训练权重，需要单独初始化。
        nn.init.normal_(self.clshead.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.clshead.bias)
        nn.init.normal_(self.bboxhead.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.bboxhead.bias)

    def predict_rois(self, features, rois):
        """把一批 RoI 的特征变成分类分数和框偏移。"""
        x = self.roi_pool(features, rois)
        x = self.flatten(x)
        x = self.classifier(x)
        cls_scores = self.clshead(x)
        bbox_deltas = self.bboxhead(x)
        return cls_scores, bbox_deltas

    def forward(self, image, rois):
        features = self.backbone(image)
        return self.predict_rois(features, rois)


if __name__ == "__main__":
    image = torch.randn(2, 3, 800, 600)
    model = FastRCNN()

    rois = torch.tensor(
        [
            [0, 50, 50, 200, 200],
            [1, 100, 100, 300, 300],
        ],
    )  #  ROI coordinates in (batch_idx, x1, y1, x2, y2) format
    cls_scores, bbox_deltas = model(image, rois)
    print(
        cls_scores.shape
    )  # Expected output shape: [N, 21] where N is the number of ROIs and 21 is the number of classes
    print(
        bbox_deltas.shape
    )  # Expected output shape: [N, 80] where N is the number of ROIs and 80 is 4 * 20
