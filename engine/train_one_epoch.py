import torch

from utils.target import (
    sample_rois,
    encode_boxes,
    getlabel,
)

from utils.loss import multi_task_loss


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    num_rois_per_image=64,
    fg_fraction=0.25,
):
    model.train()

    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    bbox_loss_sum = 0.0

    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):

        # =========================================================
        # 1. 从 DataLoader 取得一个 batch
        # =========================================================

        images = batch["images"].to(device)
        # [B, 3, H, W]

        proposals_list = batch["proposals"]
        # list:
        # [
        #   [N0, 4],
        #   [N1, 4],
        #   ...
        # ]

        gt_boxes_list = batch["gt_boxes"]
        # [
        #   [M0, 4],
        #   [M1, 4],
        #   ...
        # ]

        gt_labels_list = batch["gt_labels"]
        # [
        #   [M0],
        #   [M1],
        #   ...
        # ]

        B = images.shape[0]

        # 用来收集整个 batch 最终参与训练的 RoI
        all_rois = []
        all_labels = []
        all_bbox_targets = []

        # =========================================================
        # 2. 对 batch 中的每张图片分别处理 proposals
        # =========================================================

        for image_idx in range(B):

            proposals = proposals_list[image_idx].to(device)
            # [N, 4]

            gt_boxes = gt_boxes_list[image_idx].to(device)
            # [M, 4]

            gt_labels = gt_labels_list[image_idx].to(device).long()
            # [M]

            # -----------------------------------------------------
            # 2.1 proposal 与所有 GT 算 IoU
            #
            # 每个 proposal 找到 IoU 最大的 GT
            #
            # label:
            # >0 foreground
            #  0 background
            # -1 ignore
            # -----------------------------------------------------

            max_iou, matched_gt_idx, labels = getlabel(
                proposals,
                gt_boxes,
                gt_labels,
            )

            # -----------------------------------------------------
            # 2.2 hierarchical sampling
            #
            # 一张图最多取 64 个 RoI
            # 尽量：
            # 16 foreground
            # 48 background
            # -----------------------------------------------------

            selected_indices = sample_rois(
                labels,
                num_rois=num_rois_per_image,
                fg_fraction=fg_fraction,
            )

            # 如果这张图片没有任何可采样 proposal
            if selected_indices.numel() == 0:
                continue

            # 真正用于本次训练的 proposals
            sampled_proposals = proposals[selected_indices]
            # [R, 4]

            sampled_labels = labels[selected_indices]
            # [R]

            # -----------------------------------------------------
            # 2.3 找到这些 sampled proposals 各自对应的 GT
            # -----------------------------------------------------

            sampled_matched_gt_idx = matched_gt_idx[selected_indices]

            matched_gt_boxes = gt_boxes[sampled_matched_gt_idx]
            # [R, 4]

            # -----------------------------------------------------
            # 2.4 proposal + matched GT
            #     -> bbox regression target
            #
            # [tx, ty, tw, th]
            # -----------------------------------------------------

            bbox_targets = encode_boxes(
                sampled_proposals,
                matched_gt_boxes,
            )
            # [R, 4]

            # background 这里虽然也算出了 target，
            # 后面 bbox loss 会通过 labels > 0 忽略它们。

            # -----------------------------------------------------
            # 2.5 给每个 RoI 加上 image_idx
            #
            # 原：
            # [x1, y1, x2, y2]
            #
            # 变：
            # [batch_idx, x1, y1, x2, y2]
            # -----------------------------------------------------

            roi_batch_index = torch.full(
                (sampled_proposals.shape[0], 1),
                fill_value=image_idx,
                dtype=sampled_proposals.dtype,
                device=device,
            )

            rois = torch.cat(
                [
                    roi_batch_index,
                    sampled_proposals,
                ],
                dim=1,
            )
            # [R, 5]

            # -----------------------------------------------------
            # 收集当前图片的数据
            # -----------------------------------------------------

            all_rois.append(rois)
            all_labels.append(sampled_labels)
            all_bbox_targets.append(bbox_targets)

        # =========================================================
        # 3. 把整个 batch 中所有图片的 RoI 拼起来
        # =========================================================

        if len(all_rois) == 0:
            continue

        rois = torch.cat(all_rois, dim=0)
        # 例如 B=2:
        # [128, 5]

        labels = torch.cat(all_labels, dim=0).long()
        # [128]

        bbox_targets = torch.cat(
            all_bbox_targets,
            dim=0,
        )
        # [128, 4]

        # =========================================================
        # 4. Forward
        # =========================================================

        optimizer.zero_grad()

        cls_scores, bbox_deltas = model(
            images,
            rois,
        )

        # cls_scores:
        # [R_total, 21]

        # bbox_deltas:
        # [R_total, 80]

        # =========================================================
        # 5. Multi-task Loss
        # =========================================================

        loss, cls_loss, bbox_loss = multi_task_loss(
            cls_scores,
            bbox_deltas,
            labels,
            bbox_targets,
        )

        # =========================================================
        # 6. Backward
        # =========================================================

        loss.backward()

        optimizer.step()

        # =========================================================
        # 7. 记录 loss
        # =========================================================

        total_loss_sum += loss.item()
        cls_loss_sum += cls_loss.item()
        bbox_loss_sum += bbox_loss.item()

        num_batches += 1

        print(
            f"Batch [{batch_idx + 1}/{len(dataloader)}] "
            f"RoIs: {len(rois)} | "
            f"Loss: {loss.item():.4f} | "
            f"Cls: {cls_loss.item():.4f} | "
            f"BBox: {bbox_loss.item():.4f}"
        )

    # =============================================================
    # epoch 平均 loss
    # =============================================================

    if num_batches == 0:
        return {
            "loss": 0.0,
            "cls_loss": 0.0,
            "bbox_loss": 0.0,
        }

    return {
        "loss": total_loss_sum / num_batches,
        "cls_loss": cls_loss_sum / num_batches,
        "bbox_loss": bbox_loss_sum / num_batches,
    }
