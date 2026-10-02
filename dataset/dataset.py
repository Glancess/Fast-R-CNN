import os

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F
from torchvision.transforms import Normalize

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# 0 留给 background
CLASS_TO_IDX = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}


class FastRCNNVOCDataset(Dataset):
    def __init__(
        self,
        root,
        proposal_dir,
        year="2007",
        image_set="trainval",
        short_side=600,
        max_side=1000,
        horizontal_flip=False,
        download=False,
    ):
        super().__init__()

        self.voc = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
        )

        self.proposal_dir = proposal_dir

        self.short_side = short_side
        self.max_side = max_side
        self.horizontal_flip = horizontal_flip

        # VGG16 ImageNet normalization
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, index):

        # =====================================================
        # 1. 读取 VOC
        # =====================================================

        image, target = self.voc[index]

        annotation = target["annotation"]

        filename = annotation["filename"]
        image_id = os.path.splitext(filename)[0]

        # PIL:
        # size = (W, H)
        old_w, old_h = image.size

        # =====================================================
        # 2. 读取 GT
        # =====================================================

        objects = annotation.get("object", [])

        # 为了兼容只有一个 object 的情况
        if isinstance(objects, dict):
            objects = [objects]

        gt_boxes = []
        gt_labels = []

        for obj in objects:

            # 可以暂时跳过 difficult
            difficult = int(obj.get("difficult", 0))

            if difficult == 1:
                continue

            class_name = obj["name"]

            label = CLASS_TO_IDX[class_name]

            box = obj["bndbox"]

            xmin = float(box["xmin"])
            ymin = float(box["ymin"])
            xmax = float(box["xmax"])
            ymax = float(box["ymax"])

            # ================================================
            # VOC:
            # 1-based + inclusive
            #
            # 转成项目统一：
            # 0-based xyxy
            # ================================================

            x1 = xmin - 1
            y1 = ymin - 1
            x2 = xmax
            y2 = ymax

            gt_boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

            gt_labels.append(label)

        gt_boxes = torch.tensor(
            gt_boxes,
            dtype=torch.float32,
        ).reshape(-1, 4)

        gt_labels = torch.tensor(
            gt_labels,
            dtype=torch.long,
        )

        # =====================================================
        # 3. 读取提前生成好的 Selective Search proposals
        # =====================================================

        proposal_path = os.path.join(
            self.proposal_dir,
            image_id + ".pt",
        )

        proposals = torch.load(
            proposal_path,
            map_location="cpu",
        ).float().reshape(-1, 4)

        # proposals 已经是：
        # [x1, y1, x2, y2]
        #
        # 而且是原始图片坐标系

        # =====================================================
        # 4. 等比例 resize 图片：短边约 600，长边不超过 1000。
        # 不强行拉成 600×800，否则物体形状会变。
        # =====================================================

        scale = min(
            self.short_side / min(old_h, old_w),
            self.max_side / max(old_h, old_w),
        )
        new_h = round(old_h * scale)
        new_w = round(old_w * scale)

        image = F.resize(
            image,
            [new_h, new_w],
        )

        # =====================================================
        # 5. GT / proposal 跟着 resize
        # =====================================================

        scale_x = new_w / old_w
        scale_y = new_h / old_h

        # GT
        if len(gt_boxes) > 0:
            gt_boxes[:, [0, 2]] *= scale_x
            gt_boxes[:, [1, 3]] *= scale_y

        # proposal
        if len(proposals) > 0:
            proposals[:, [0, 2]] *= scale_x
            proposals[:, [1, 3]] *= scale_y

        # =====================================================
        # 6. 限制坐标范围
        # =====================================================

        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clamp(0, new_w)

        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clamp(0, new_h)

        proposals[:, [0, 2]] = proposals[:, [0, 2]].clamp(0, new_w)

        proposals[:, [1, 3]] = proposals[:, [1, 3]].clamp(0, new_h)

        # =====================================================
        # 7. 过滤非法 proposal
        # =====================================================

        widths = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]

        valid = (widths > 1) & (heights > 1)

        proposals = proposals[valid]

        # 只在训练集随机左右翻转。图片、GT 和 proposals 必须一起翻转。
        if self.horizontal_flip and torch.rand(1).item() < 0.5:
            image = F.hflip(image)
            if len(gt_boxes) > 0:
                old_x1 = gt_boxes[:, 0].clone()
                old_x2 = gt_boxes[:, 2].clone()
                gt_boxes[:, 0] = new_w - old_x2
                gt_boxes[:, 2] = new_w - old_x1
            if len(proposals) > 0:
                old_x1 = proposals[:, 0].clone()
                old_x2 = proposals[:, 2].clone()
                proposals[:, 0] = new_w - old_x2
                proposals[:, 2] = new_w - old_x1

        # =====================================================
        # 8. PIL -> Tensor + VGG normalization
        # =====================================================

        image = F.to_tensor(image)

        image = self.normalize(image)

        return {
            "image": image,
            "proposals": proposals,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "image_id": image_id,
        }


# =============================================================
# Detection 专用 collate
# =============================================================


def detection_collate_fn(batch):

    # 等比例缩放后图片大小可能不同，补 0 后才能组成一个 batch。
    # 补的 0 是归一化后的 0，不影响真实图片中的坐标。
    max_h = max(item["image"].shape[1] for item in batch)
    max_w = max(item["image"].shape[2] for item in batch)
    images = torch.zeros(len(batch), 3, max_h, max_w)
    for i, item in enumerate(batch):
        image = item["image"]
        images[i, :, : image.shape[1], : image.shape[2]] = image

    # proposals / GT 数量每张图不同
    # 所以保持 list
    proposals = [item["proposals"] for item in batch]

    gt_boxes = [item["gt_boxes"] for item in batch]

    gt_labels = [item["gt_labels"] for item in batch]

    image_ids = [item["image_id"] for item in batch]

    return {
        "images": images,
        "proposals": proposals,
        "gt_boxes": gt_boxes,
        "gt_labels": gt_labels,
        "image_ids": image_ids,
    }


if __name__ == "__main__":

    dataset = FastRCNNVOCDataset(
        root="./data",
        proposal_dir="./data/VOCdevkit/VOC2007/SelectiveSearchProposals",
        year="2007",
        image_set="trainval",
        short_side=600,
        max_side=1000,
    )

    print("dataset:", len(dataset))

    sample = dataset[0]

    print("image:", sample["image"].shape)
    print("proposals:", sample["proposals"].shape)
    print("gt_boxes:", sample["gt_boxes"])
    print("gt_labels:", sample["gt_labels"])
    print("image_id:", sample["image_id"])
