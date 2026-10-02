import torch

from utils.loss import multi_task_loss
from utils.target import encode_boxes, getlabel, sample_rois


@torch.no_grad()
def evaluate_one_epoch(model, dataloader, device, num_rois_per_image=64):
    """在固定抽取的 val RoI 上计算 loss；这不是目标检测 mAP。"""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        all_rois = []
        all_labels = []
        all_bbox_targets = []

        for image_idx in range(len(images)):
            proposals = batch["proposals"][image_idx].to(device)
            gt_boxes = batch["gt_boxes"][image_idx].to(device)
            gt_labels = batch["gt_labels"][image_idx].to(device).long()
            if len(proposals) == 0 or len(gt_boxes) == 0:
                continue

            # 和训练时一样匹配 GT，但固定选择 RoI，避免每次 val 随机变化。
            _, matched_gt_idx, labels = getlabel(proposals, gt_boxes, gt_labels)
            selected = sample_rois(
                labels,
                num_rois=num_rois_per_image,
                fg_fraction=0.25,
                random_sample=False,
            )
            if len(selected) == 0:
                continue

            sampled_proposals = proposals[selected]
            sampled_labels = labels[selected]
            matched_gt_boxes = gt_boxes[matched_gt_idx[selected]]
            bbox_targets = encode_boxes(sampled_proposals, matched_gt_boxes)

            # 第一列指出这个 RoI 属于 batch 中的第几张图片。
            image_column = torch.full(
                (len(selected), 1),
                image_idx,
                dtype=sampled_proposals.dtype,
                device=device,
            )
            rois = torch.cat([image_column, sampled_proposals], dim=1)
            all_rois.append(rois)
            all_labels.append(sampled_labels)
            all_bbox_targets.append(bbox_targets)

        if len(all_rois) == 0:
            continue

        rois = torch.cat(all_rois, dim=0)
        labels = torch.cat(all_labels, dim=0).long()
        bbox_targets = torch.cat(all_bbox_targets, dim=0)
        cls_scores, bbox_deltas = model(images, rois)
        loss, cls_loss, bbox_loss = multi_task_loss(
            cls_scores, bbox_deltas, labels, bbox_targets
        )

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_bbox_loss += bbox_loss.item()
        num_batches += 1

        if (batch_idx + 1) % 100 == 0 or batch_idx + 1 == len(dataloader):
            print(f"Val batch [{batch_idx + 1}/{len(dataloader)}]")

    if num_batches == 0:
        raise ValueError("val 没有可用的 RoI，请检查 GT 和 proposal")

    return {
        "loss": total_loss / num_batches,
        "cls_loss": total_cls_loss / num_batches,
        "bbox_loss": total_bbox_loss / num_batches,
    }
