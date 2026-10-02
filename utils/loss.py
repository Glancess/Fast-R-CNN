import torch
import torch.nn.functional as F


def multi_task_loss(cls_scores, bbox_deltas, labels, bbox_targets):

    # 1. 分类：所有 sampled RoI 都参与
    cls_loss = F.cross_entropy(cls_scores, labels)

    # [N, 80] -> [N, 20, 4]
    bbox_deltas = bbox_deltas.view(-1, 20, 4)

    # 2. 找 foreground
    pos_indices = torch.where(labels > 0)[0]

    if len(pos_indices) > 0:

        # foreground 的真实类别
        pos_labels = labels[pos_indices]
        # [num_pos]

        # foreground 的所有 class-specific bbox predictions
        bbox_deltas_pos = bbox_deltas[pos_indices]
        # [num_pos, 20, 4]

        # 每个 foreground 只取自己类别对应的 4 个
        class_indices = pos_labels - 1

        pred_bbox = bbox_deltas_pos[
            torch.arange(len(pos_indices), device=bbox_deltas.device), class_indices
        ]
        # [num_pos, 4]

        bbox_targets_pos = bbox_targets[pos_indices]
        # [num_pos, 4]

        bbox_loss = F.smooth_l1_loss(pred_bbox, bbox_targets_pos)

    else:
        bbox_loss = bbox_deltas.sum() * 0.0

    total_loss = cls_loss + bbox_loss
    return total_loss, cls_loss, bbox_loss


if __name__ == "__main__":
    # Example usage
    cls_scores = torch.randn(10, 21)  # 10 samples, 21 classes
    bbox_deltas = torch.randn(
        10, 4 * 20
    )  # 10 samples, 4 deltas for each of the 20 classes
    labels = torch.randint(0, 21, (10,))  # Random labels between 0 and 20
    bbox_targets = torch.randn(10, 4)  # Ground truth bounding boxes

    total_loss, cls_loss, bbox_loss = multi_task_loss(
        cls_scores, bbox_deltas, labels, bbox_targets
    )
    print("Classification Loss:", cls_loss.item())
    print("Bounding Box Loss:", bbox_loss.item())
    print("Total Loss:", total_loss.item())
