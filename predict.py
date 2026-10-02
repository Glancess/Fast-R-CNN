"""用 train-only 模型在一张 VOC val 图片上做检测和画框。"""

import sys

import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms
from torchvision.transforms.functional import to_pil_image

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


def save_compare_image(sample, detections):
    """左边画真实框，右边画预测框，保存一张对比图。"""
    # Dataset 已把图片归一化；画图前先恢复到 0～1 的 RGB 范围。
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (sample["image"].cpu() * std + mean).clamp(0, 1)
    image = to_pil_image(image)

    gt_image = image.copy()
    pred_image = image.copy()
    gt_draw = ImageDraw.Draw(gt_image)
    pred_draw = ImageDraw.Draw(pred_image)

    # GT、预测框都使用 Dataset 缩放后的坐标，和这里的图片尺寸一致。
    for box, label in zip(sample["gt_boxes"], sample["gt_labels"]):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        name = VOC_CLASSES[label.item() - 1]
        gt_draw.rectangle((x1, y1, x2, y2), outline="lime", width=3)
        gt_draw.text((x1, max(0, y1 - 12)), name, fill="lime")

    for detection in detections:
        x1, y1, x2, y2 = [int(v) for v in detection["box"]]
        name = detection["class"]
        score = detection["score"]
        pred_draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        pred_draw.text(
            (x1, max(0, y1 - 12)), f"{name} {score:.2f}", fill="red"
        )

    width, height = image.size
    compare = Image.new("RGB", (width * 2, height + 24), "white")
    compare.paste(gt_image, (0, 24))
    compare.paste(pred_image, (width, 24))
    title_draw = ImageDraw.Draw(compare)
    title_draw.text((5, 5), "Ground Truth", fill="green")
    title_draw.text((width + 5, 5), "Prediction", fill="red")

    output_path = CHECKPOINT_DIR / f"{sample['image_id']}_compare.jpg"
    compare.save(output_path)
    return output_path


def main():
    if len(sys.argv) != 2:
        raise ValueError("用法：python predict.py 图片序号，例如 python predict.py 0")

    image_index = int(sys.argv[1])
    checkpoint_path = CHECKPOINT_DIR / "best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"还没有训练好的 checkpoint：{checkpoint_path}")

    dataset = FastRCNNVOCDataset(
        root=str(DATA_ROOT),
        proposal_dir=str(PROPOSAL_DIR),
        image_set="val",
    )
    sample = dataset[image_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastRCNN(pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("train_set") != "train":
        raise ValueError("这个 checkpoint 不是 train-only 训练的，不能用于 val 评估")
    model.load_state_dict(checkpoint["model"])

    print("image_set: val, image_id:", sample["image_id"])
    detections = predict_one(
        model, sample["image"], sample["proposals"], device
    )
    for detection in detections[:20]:
        print(detection)
    output_path = save_compare_image(sample, detections)
    print("对比图已保存：", output_path)


if __name__ == "__main__":
    main()
