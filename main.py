import torch
from torch.utils.data import Dataset, DataLoader

from model import FastRCNN
from engine.train_one_epoch import train_one_epoch

# ============================================================
# 1. 人工构造一个假的检测数据集
# ============================================================


class FakeDetectionDataset(Dataset):
    def __init__(self):
        self.num_images = 2

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        # 为了单元测试先用较小图片
        image = torch.randn(3, 400, 400)

        if idx == 0:

            # ------------------------------------------------
            # 图片 0：两个 GT
            # ------------------------------------------------
            gt_boxes = torch.tensor(
                [
                    [100, 100, 300, 300],
                    [250, 50, 380, 200],
                ],
                dtype=torch.float32,
            )

            # 假设：
            # 1 = 类别1
            # 5 = 类别5
            gt_labels = torch.tensor(
                [1, 5],
                dtype=torch.long,
            )

            # ------------------------------------------------
            # 人工 proposals
            #
            # 有：
            # - foreground
            # - background
            # - ignore
            # ------------------------------------------------
            proposals = torch.tensor(
                [
                    # foreground：与 GT0 高度重合
                    [100, 100, 300, 300],
                    [110, 110, 290, 290],
                    # foreground：与 GT1 高度重合
                    [250, 50, 380, 200],
                    [260, 60, 370, 190],
                    # background：和 GT 有一些重合
                    [50, 50, 250, 250],
                    [150, 150, 350, 350],
                    [200, 20, 320, 130],
                    [300, 100, 399, 250],
                    # ignore：几乎完全没关系
                    [0, 0, 40, 40],
                    [10, 320, 80, 390],
                ],
                dtype=torch.float32,
            )

        else:

            # ------------------------------------------------
            # 图片 1
            # ------------------------------------------------
            gt_boxes = torch.tensor(
                [
                    [50, 150, 200, 350],
                    [220, 120, 360, 300],
                ],
                dtype=torch.float32,
            )

            gt_labels = torch.tensor(
                [3, 10],
                dtype=torch.long,
            )

            proposals = torch.tensor(
                [
                    # foreground
                    [50, 150, 200, 350],
                    [60, 160, 190, 340],
                    [220, 120, 360, 300],
                    [230, 130, 350, 290],
                    # background
                    [20, 100, 160, 280],
                    [100, 180, 260, 370],
                    [180, 80, 300, 220],
                    [280, 100, 399, 260],
                    # ignore
                    [0, 0, 30, 30],
                    [330, 330, 390, 390],
                ],
                dtype=torch.float32,
            )

        return {
            "image": image,
            "proposals": proposals,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
        }


# ============================================================
# 2. 自定义 collate_fn
#
# 因为：
# 每张图 proposal 数量可能不同
# 每张图 GT 数量也可能不同
#
# 所以不能全部 torch.stack()
# ============================================================


def detection_collate_fn(batch):

    images = torch.stack(
        [item["image"] for item in batch],
        dim=0,
    )

    proposals_list = [item["proposals"] for item in batch]

    gt_boxes_list = [item["gt_boxes"] for item in batch]

    gt_labels_list = [item["gt_labels"] for item in batch]

    return {
        "images": images,
        "proposals": proposals_list,
        "gt_boxes": gt_boxes_list,
        "gt_labels": gt_labels_list,
    }


# ============================================================
# 3. 开始测试
# ============================================================

if __name__ == "__main__":

    # Mac 可以用 MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("device:", device)

    dataset = FakeDetectionDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=detection_collate_fn,
    )

    model = FastRCNN().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-4,
        momentum=0.9,
    )

    result = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        # 测试阶段不要一张图采 64 个
        # 我们总共才人工造了 10 个 proposal
        num_rois_per_image=8,
        # 25% foreground
        fg_fraction=0.25,
    )

    print()
    print("===== Epoch Result =====")

    print(f"Total Loss: {result['loss']:.4f}")

    print(f"Cls Loss: {result['cls_loss']:.4f}")

    print(f"BBox Loss: {result['bbox_loss']:.4f}")
