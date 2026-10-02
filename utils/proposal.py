# proposal.py
import cv2
import torch


def selective_search(image_bgr, mode="fast", max_proposals=2000):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    ss.setBaseImage(image_bgr)

    if mode == "quality":
        ss.switchToSelectiveSearchQuality()
    else:
        ss.switchToSelectiveSearchFast()

    rects = ss.process()
    # rects 每个是 (x, y, w, h)

    proposals = []

    for x, y, w, h in rects[:max_proposals]:
        proposals.append([x, y, x + w, y + h])

    return torch.tensor(proposals, dtype=torch.float32)


if __name__ == "__main__":
    # 测试 selective_search
    image_bgr = cv2.imread("test.jpg")
    proposals = selective_search(image_bgr, mode="fast", max_proposals=2000)
    print(proposals.shape)  # Expected output: [N, 4] where N is the number of proposals
