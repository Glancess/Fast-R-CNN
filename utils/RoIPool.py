import math
import torch
import torch.nn as nn
from utils.map_roi_to_feature_map import map_roi_to_feature_map


class RoIPool(nn.Module):
    def __init__(self, output_size=(7, 7), spatial_scale=1 / 16):
        super().__init__()
        self.output_h = output_size[0]
        self.output_w = output_size[1]
        self.spatial_scale = spatial_scale

    def forward(self, feature, rois):
        """
        feature: [B, C, H, W]

        rois: [N, 5]
        每一行：
        [batch_idx, x1, y1, x2, y2]

        注意 rois 坐标是原图坐标
        """

        outputs = []

        for roi in rois:
            batch_idx = int(roi[0].item())

            # 1. 原图坐标 -> feature map 坐标
            x1, y1, x2, y2 = map_roi_to_feature_map(
                (roi[1].item(), roi[2].item(), roi[3].item(), roi[4].item()),
                self.spatial_scale,
            )

            roi_w = x2 - x1
            roi_h = y2 - y1

            bin_w = roi_w / self.output_w
            bin_h = roi_h / self.output_h

            pooled = torch.zeros(
                feature.shape[1],
                self.output_h,
                self.output_w,
                device=feature.device,
                dtype=feature.dtype,
            )  # 用来存放每个 ROI 的池化结果

            for i in range(self.output_h):
                for j in range(self.output_w):

                    xs = math.floor(x1 + j * bin_w)
                    xe = math.ceil(x1 + (j + 1) * bin_w)
                    ys = math.floor(y1 + i * bin_h)
                    ye = math.ceil(y1 + (i + 1) * bin_h)
                    # 防止越界
                    xs = max(0, min(xs, feature.shape[3]))
                    xe = max(0, min(xe, feature.shape[3]))
                    ys = max(0, min(ys, feature.shape[2]))
                    ye = max(0, min(ye, feature.shape[2]))

                    region = feature[batch_idx, :, ys:ye, xs:xe]
                    if region.numel() > 0:
                        pooled[:, i, j] = region.amax(dim=(-2, -1))
            outputs.append(pooled)
        return torch.stack(outputs, dim=0)
