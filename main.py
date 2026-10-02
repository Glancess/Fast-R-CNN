import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from dataset.dataset import FastRCNNVOCDataset
from dataset.dataset import detection_collate_fn as voc_collate_fn
from model import FastRCNN
from engine.train_one_epoch import train_one_epoch
from engine.evaluate_one_epoch import evaluate_one_epoch


# 先保持最基础的单机训练设置，方便逐项理解。
PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_DIR / "dataset" / "data"
PROPOSAL_DIR = DATA_ROOT / "VOCdevkit" / "VOC2007" / "SelectiveSearchProposals"
# 和以前 trainval 训练得到的 checkpoints/last.pth 分开存放。
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / "train_only"
EPOCHS = 10
BATCH_SIZE = 2
NUM_ROIS_PER_IMAGE = 64
LEARNING_RATE = 0.001
RESUME = True  # 有 last.pth 时继续训练；想从头开始可改成 False

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

def run_fake_demo():

    # torchvision 的 RoI Pool 算子在 CUDA / CPU 上运行。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def train_voc():
    # 只用 train 更新参数，val 只用于验证。proposal 文件仍可沿用 trainval 生成的。
    train_dataset = FastRCNNVOCDataset(
        root=str(DATA_ROOT),
        proposal_dir=str(PROPOSAL_DIR),
        image_set="train",
        horizontal_flip=True,
    )
    val_dataset = FastRCNNVOCDataset(
        root=str(DATA_ROOT),
        proposal_dir=str(PROPOSAL_DIR),
        image_set="val",
        horizontal_flip=False,
    )

    train_ids = {Path(path).stem for path in train_dataset.voc.images}
    val_ids = {Path(path).stem for path in val_dataset.voc.images}
    if train_ids & val_ids:
        raise ValueError("train 和 val 有重复图片，请检查划分文件")

    # train 和 val 都需要 proposal；启动前检查，避免训练到验证时才报错。
    for split_name, split_dataset in (("train", train_dataset), ("val", val_dataset)):
        missing = []
        for image_path in split_dataset.voc.images:
            image_id = Path(image_path).stem
            if not (PROPOSAL_DIR / (image_id + ".pt")).is_file():
                missing.append(image_id)
        if missing:
            raise FileNotFoundError(
                f"{split_name} 还缺 {len(missing)} 个 proposal 文件。"
                f"例如 {missing[:5]}。请等服务器生成完再训练。"
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=voc_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=voc_collate_fn,
    )

    # RoI Pool 的现成算子在 CUDA / CPU 上运行；服务器优先使用 CUDA。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("train images:", len(train_dataset))
    print("val images:", len(val_dataset))

    last_path = CHECKPOINT_DIR / "last.pth"
    best_path = CHECKPOINT_DIR / "best.pth"
    resuming = RESUME and last_path.is_file()
    # 续训会立刻加载自己的权重，不必再下载 ImageNet 权重。
    model = FastRCNN(pretrained=not resuming).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0005,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_val_loss = float("inf")
    if resuming:
        checkpoint = torch.load(last_path, map_location=device)
        if checkpoint.get("train_set") != "train":
            raise ValueError("这个 checkpoint 不是 train-only 训练的，不能用来续训")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"从第 {start_epoch + 1} 轮继续训练")

    if start_epoch >= EPOCHS:
        print("checkpoint 已经完成设定的轮数；想继续训练请增大 EPOCHS。")
        return

    for epoch in range(start_epoch, EPOCHS):
        train_result = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            num_rois_per_image=NUM_ROIS_PER_IMAGE,
            fg_fraction=0.25,
        )
        val_result = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            num_rois_per_image=NUM_ROIS_PER_IMAGE,
        )
        print(f"Epoch [{epoch + 1}/{EPOCHS}] train: {train_result}")
        print(f"Epoch [{epoch + 1}/{EPOCHS}] val:   {val_result}")

        # best 按 val loss 选；last 是最新一轮，用于续训。
        is_best = val_result["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_result["loss"]
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_result["loss"],
            "best_val_loss": best_val_loss,
            "train_set": "train",
        }
        torch.save(checkpoint, last_path)
        if is_best:
            torch.save(checkpoint, best_path)
            print("val loss 降低，已保存 best.pth")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--fake":
        run_fake_demo()
    elif len(sys.argv) == 1:
        train_voc()
    else:
        raise ValueError("用法：python main.py  或  python main.py --fake")
