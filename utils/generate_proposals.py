import os
import cv2
import torch

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

VOC_ROOT = "/root/Fast-R-CNN/dataset/data/VOCdevkit/VOC2007"

IMAGE_DIR = os.path.join(
    VOC_ROOT,
    "JPEGImages",
)

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

os.makedirs(
    SAVE_DIR,
    exist_ok=True,
)


def selective_search(image_bgr, max_proposals=2000):

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    ss.setBaseImage(image_bgr)

    # 使用快速模式
    ss.switchToSelectiveSearchFast()

    rects = ss.process()

    proposals = []

    for x, y, w, h in rects:

        if w <= 0 or h <= 0:
            continue

        # OpenCV:
        # (x, y, w, h)
        #
        # 项目统一：
        # (x1, y1, x2, y2)

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


def process_one_image(image_id):

    # 很重要：
    # 每个 Python 进程内部不要再让 OpenCV 开一堆线程
    cv2.setNumThreads(1)

    image_path = os.path.join(
        IMAGE_DIR,
        image_id + ".jpg",
    )

    save_path = os.path.join(
        SAVE_DIR,
        image_id + ".pt",
    )

    # 已经生成过的直接跳过
    if os.path.exists(save_path):
        return True

    image = cv2.imread(image_path)

    if image is None:
        print("Failed:", image_path)
        return False

    proposals = selective_search(
        image,
        max_proposals=2000,
    )

    torch.save(
        proposals,
        save_path,
    )

    return True


def main():

    with open(IMAGE_SET_FILE, "r") as f:

        image_ids = [line.strip() for line in f if line.strip()]

    print(f"Total images: {len(image_ids)}")

    num_workers = 12

    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        results = executor.map(
            process_one_image,
            image_ids,
        )

        for _ in tqdm(
            results,
            total=len(image_ids),
        ):
            pass


if __name__ == "__main__":
    main()
