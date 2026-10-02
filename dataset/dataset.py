import os

import torch
from torch.utils.data import Dataset, DataLoader
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
        image_size=(600, 800),  # (H, W)
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

        self.new_h = image_size[0]
        self.new_w = image_size[1]  # H=600, W=800

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

        objects = annotation["object"]

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
        ).float()

        # proposals 已经是：
        # [x1, y1, x2, y2]
        #
        # 而且是原始图片坐标系

        # =====================================================
        # 4. resize 图片
        # =====================================================

        image = F.resize(
            image,
            [self.new_h, self.new_w],
        )

        # =====================================================
        # 5. GT / proposal 跟着 resize
        # =====================================================

        scale_x = self.new_w / old_w
        scale_y = self.new_h / old_h

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

        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clamp(0, self.new_w)

        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clamp(0, self.new_h)

        proposals[:, [0, 2]] = proposals[:, [0, 2]].clamp(0, self.new_w)

        proposals[:, [1, 3]] = proposals[:, [1, 3]].clamp(0, self.new_h)

        # =====================================================
        # 7. 过滤非法 proposal
        # =====================================================

        widths = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]

        valid = (widths > 1) & (heights > 1)

        proposals = proposals[valid]

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

    # 因为我们统一 resize 成 600×800，
    # image 可以直接 stack

    images = torch.stack(
        [item["image"] for item in batch],
        dim=0,
    )

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
        image_size=(600, 800),
    )

    print("dataset:", len(dataset))

    sample = dataset[0]

    print("image:", sample["image"].shape)
    print("proposals:", sample["proposals"].shape)
    print("gt_boxes:", sample["gt_boxes"])
    print("gt_labels:", sample["gt_labels"])
    print("image_id:", sample["image_id"])
