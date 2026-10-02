import os

import cv2
import torch
from tqdm import tqdm

VOC_ROOT = "projects/Fast-R-CNN/dataset/data/VOCdevkit/VOC2007"

IMAGE_DIR = os.path.join(VOC_ROOT, "JPEGImages")

IMAGE_SET_FILE = os.path.join(
    VOC_ROOT,
    "ImageSets",
    "Main",
    "trainval.txt",
)

SAVE_DIR = os.path.join(
    VOC_ROOT,
    "SelectiveSearchProposals",
)

os.makedirs(SAVE_DIR, exist_ok=True)


def selective_search(image_bgr, max_proposals=2000):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    ss.setBaseImage(image_bgr)

    # 先用 fast
    ss.switchToSelectiveSearchFast()

    rects = ss.process()

    proposals = []

    for x, y, w, h in rects:

        if w <= 0 or h <= 0:
            continue

        # OpenCV:
        # xywh
        #
        # 我们项目统一：
        # xyxy
        proposals.append(
            [
                float(x),
                float(y),
                float(x + w),
                float(y + h),
            ]
        )

        if len(proposals) >= max_proposals:
            break

    return torch.tensor(
        proposals,
        dtype=torch.float32,
    )


def main():

    with open(IMAGE_SET_FILE, "r") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    print(f"Total images: {len(image_ids)}")

    for image_id in tqdm(image_ids):

        image_path = os.path.join(
            IMAGE_DIR,
            image_id + ".jpg",
        )

        save_path = os.path.join(
            SAVE_DIR,
            image_id + ".pt",
        )

        # 已经处理过就跳过
        if os.path.exists(save_path):
            continue

        image = cv2.imread(image_path)

        if image is None:
            print("Failed:", image_path)
            continue

        proposals = selective_search(
            image,
            max_proposals=2000,
        )

        torch.save(
            proposals,
            save_path,
        )


if __name__ == "__main__":
    main()
