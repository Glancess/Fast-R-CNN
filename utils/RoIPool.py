import torch.nn as nn
from torchvision.ops import roi_pool


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

        if len(rois) == 0:
            return feature.new_zeros(
                (0, feature.shape[1], self.output_h, self.output_w)
            )

        # 和手写版本做同一件事：每个 RoI 池化为 7×7。
        # 现成算子避免逐个 RoI、逐个格子运行 Python 循环，也支持反向传播。
        return roi_pool(
            feature,
            rois,
            output_size=(self.output_h, self.output_w),
            spatial_scale=self.spatial_scale,
        )
