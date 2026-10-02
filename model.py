import math

import torch
import torch.nn as nn
from torchvision.models import vgg16
from utils.RoIPool import RoIPool


class FastRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True)

        self.backbone = vgg.features[:-1]
        self.classifier = vgg.classifier[:-1]

        self.roi_pool = RoIPool()

        self.clshead = nn.Linear(4096, 21)
        self.bboxhead = nn.Linear(4096, 4 * 20)
        self.flatten = nn.Flatten()

    def forward(self, image, rois):
        x = self.backbone(image)
        print("Backbone output shape:", x.shape)  # Debugging line
        x = self.roi_pool(x, rois)
        print("RoI Pool output shape:", x.shape)  # Debugging line
        x = self.flatten(x)
        print("Flatten output shape:", x.shape)  # Debugging line
        x = self.classifier(x)
        print("Classifier output shape:", x.shape)  # Debugging line
        cls_scores = self.clshead(x)
        bbox_deltas = self.bboxhead(x)
        return cls_scores, bbox_deltas


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
