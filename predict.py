"""用训练好的 last.pth 在一张 VOC 图片上做检测。"""

import sys

import torch
from torchvision.ops import nms

from dataset.dataset import FastRCNNVOCDataset, VOC_CLASSES
from main import DATA_ROOT, PROPOSAL_DIR, CHECKPOINT_DIR
from model import FastRCNN


def decode_boxes(proposals, deltas):
    """把模型预测的 [dx, dy, dw, dh] 转回图片坐标。"""
    widths = proposals[:, 2] - proposals[:, 0]
    heights = proposals[:, 3] - proposals[:, 1]
    center_x = (proposals[:, 0] + proposals[:, 2]) / 2
    center_y = (proposals[:, 1] + proposals[:, 3]) / 2

    new_x = deltas[:, 0] * widths + center_x
    new_y = deltas[:, 1] * heights + center_y
    new_w = torch.exp(deltas[:, 2].clamp(max=4)) * widths
    new_h = torch.exp(deltas[:, 3].clamp(max=4)) * heights

    return torch.stack(
        [
            new_x - new_w / 2,
            new_y - new_h / 2,
            new_x + new_w / 2,
            new_y + new_h / 2,
        ],
        dim=1,
    )


@torch.no_grad()
def predict_one(model, image, proposals, device, score_threshold=0.5):
    """返回检测框、类别和分数；背景类别 0 不输出。"""
    model.eval()
    image = image.unsqueeze(0).to(device)
    proposals = proposals.to(device)
    if len(proposals) == 0:
        return []

    # 整张图片只经过一次 VGG；RoI 分批处理，避免 2000 个 RoI 一起占满显存。
    features = model.backbone(image)
    score_parts = []
    delta_parts = []
    for start in range(0, len(proposals), 128):
        part = proposals[start : start + 128]
        batch_column = torch.zeros(len(part), 1, device=device)
        rois = torch.cat([batch_column, part], dim=1)
        scores, deltas = model.predict_rois(features, rois)
        score_parts.append(scores)
        delta_parts.append(deltas)

    probabilities = torch.cat(score_parts).softmax(dim=1)
    deltas = torch.cat(delta_parts).reshape(-1, 20, 4)
    height, width = image.shape[-2:]
    results = []

    for label in range(1, 21):
        scores = probabilities[:, label]
        selected = scores >= score_threshold
        if not selected.any():
            continue

        boxes = decode_boxes(proposals[selected], deltas[selected, label - 1])
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height)
        class_scores = scores[selected]

        # 同一类高度重叠的框只保留分数最高的一个。
        kept = nms(boxes, class_scores, iou_threshold=0.3)
        for index in kept.tolist():
            results.append(
                {
                    "class": VOC_CLASSES[label - 1],
                    "score": class_scores[index].item(),
                    "box": [round(value, 1) for value in boxes[index].tolist()],
                }
            )

    results.sort(key=lambda result: result["score"], reverse=True)
    return results


def main():
    if len(sys.argv) != 2:
        raise ValueError("用法：python predict.py 图片序号，例如 python predict.py 0")

    image_index = int(sys.argv[1])
    checkpoint_path = CHECKPOINT_DIR / "last.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"还没有训练好的 checkpoint：{checkpoint_path}")

    dataset = FastRCNNVOCDataset(
        root=str(DATA_ROOT),
        proposal_dir=str(PROPOSAL_DIR),
        image_set="trainval",
    )
    sample = dataset[image_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastRCNN(pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    print("image_id:", sample["image_id"])
    detections = predict_one(
        model, sample["image"], sample["proposals"], device
    )
    for detection in detections[:20]:
        print(detection)


if __name__ == "__main__":
    main()
