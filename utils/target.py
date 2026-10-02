import torch
from torchvision.ops import box_iou


def getlabel(proposals, gt_boxes, gt_labels):
    """
    proposals: [N, 4]
    gt_boxes:  [M, 4]
    gt_labels: [M]
    坐标格式均为:
    [x1, y1, x2, y2]
    """
    ious = box_iou(proposals, gt_boxes)  # n x m
    max_iou, matched_gt_idx = ious.max(dim=1)
    # 沿着某一维压缩”意味着那一维会被消掉。
    labels = gt_labels[matched_gt_idx]

    bg_mask = (max_iou >= 0.1) & (max_iou < 0.5)
    ignore_mask = max_iou < 0.1

    labels[bg_mask] = 0

    # -1 表示 ignore，不参加训练
    labels[ignore_mask] = -1

    return max_iou, matched_gt_idx, labels


def sample_rois(labels, num_rois=64, fg_fraction=0.25):
    """
    labels:
        > 0 : foreground
        = 0 : background
        = -1: ignore

    return:
        selected_indices
    """
    fg_indices = torch.where(labels > 0)[0]  # foreground indices
    bg_indices = torch.where(labels == 0)[0]  # background indices

    num_fg = int(num_rois * fg_fraction)
    num_fg = min(num_fg, len(fg_indices))
    num_bg = num_rois - num_fg
    num_bg = min(num_bg, len(bg_indices))

    selected_fg_indices = (
        fg_indices[torch.randperm(len(fg_indices))[:num_fg]]
        if num_fg > 0
        else torch.tensor([], dtype=torch.long)
    )
    selected_bg_indices = (
        bg_indices[torch.randperm(len(bg_indices))[:num_bg]]
        if num_bg > 0
        else torch.tensor([], dtype=torch.long)
    )
    print("Selected foreground indices:", selected_fg_indices)  # Debugging line
    print("Selected background indices:", selected_bg_indices)  # Debugging line
    selected_indices = torch.cat([selected_fg_indices, selected_bg_indices])
    return selected_indices


def encode_boxes(proposals, gt_boxes):
    """
    proposals: [N, 4]
    gt_boxes:  [N, 4]

    两者一一对应

    return:
        targets: [N, 4]
    """
    proposals = proposals.float()
    gt_boxes = gt_boxes.float()

    px = (proposals[:, 0] + proposals[:, 2]) / 2.0  # x1+x2/2
    py = (proposals[:, 1] + proposals[:, 3]) / 2.0
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]

    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]

    targets_dx = (gx - px) / pw
    targets_dy = (gy - py) / ph
    targets_dw = torch.log(gw / pw)
    targets_dh = torch.log(gh / ph)

    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


if __name__ == "__main__":
    proposals = torch.tensor([[10, 10, 20, 20], [15, 15, 25, 25], [30, 30, 40, 40]])
    gt_boxes = torch.tensor([[12, 12, 22, 22], [35, 35, 45, 45]])
    gt_labels = torch.tensor([1, 2])

    max_iou, matched_gt_idx, labels = getlabel(proposals, gt_boxes, gt_labels)
    print(max_iou)
    print(matched_gt_idx)
    print(labels)
    selected_indices = sample_rois(labels)
    print(selected_indices)
    targets = encode_boxes(
        proposals[selected_indices], gt_boxes[matched_gt_idx[selected_indices]]
    )
    print(targets)
