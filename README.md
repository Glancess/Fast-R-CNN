# Fast R-CNN（VOC 2007）

本项目用预训练 VGG16、Selective Search proposals、RoI Pool、分类头和类别专属的框回归头训练 Fast R-CNN。代码保留了原来的两张假图片演示；真实训练现在使用 `dataset/dataset.py` 中的 VOC Dataset。

## 数据准备

项目目录下需要有：

```text
dataset/data/VOCdevkit/VOC2007/
├── Annotations/
├── JPEGImages/
├── ImageSets/Main/trainval.txt
└── SelectiveSearchProposals/
    ├── 000005.pt
    └── ...
```

服务器正在运行的 `utils/generate_proposals.py` 会为 `trainval.txt` 中每张图保存一个同名 `.pt` 文件，每个文件是 `[N, 4]` 的张量，四列为原图坐标 `[x1, y1, x2, y2]`。训练代码直接读取这个格式，没有修改生成脚本。**请等所有 proposal 文件生成完再开始真实训练。** `main.py` 会在加载 VGG16 前检查是否缺文件，并列出缺失数量。

如果服务器上的项目目录是 `/root/Fast-R-CNN`，现有生成脚本中的 `VOC_ROOT` 与此目录相符；若服务器路径不同，生成脚本中的绝对路径需要由你按实际位置调整。不要在生成任务运行时改动它。

## 运行

在项目目录运行：

```bash
python main.py --fake    # 原来的两张假图片演示
python main.py           # VOC 2007 trainval 真实训练
python predict.py 0      # 用 last.pth 检测 trainval 中第 0 张图
```

训练参数直接放在 `main.py` 顶部：`EPOCHS`、`BATCH_SIZE`、`NUM_ROIS_PER_IMAGE`、`LEARNING_RATE`。这样修改时可以清楚看到每个参数。每轮结束保存 `checkpoints/last.pth`，包含模型、优化器和轮数；再次运行默认从它继续。想重新开始，可把 `RESUME` 设为 `False`，并另行保存旧 checkpoint，以免被新训练覆盖。

第一次建立模型需要 torchvision 的 ImageNet VGG16 预训练权重；若服务器尚未缓存，torchvision 会尝试下载。`predict.py` 直接加载训练 checkpoint，不会再下载预训练权重。

## 一张图片怎样参与训练

1. Dataset 读取 VOC 图片、标注框和对应的 proposal 文件。
2. 图片按比例缩放，短边约 600、长边不超过 1000；GT 和 proposals 用相同的比例缩放。训练时图片和框还会一起随机左右翻转。
3. DataLoader 把不同大小的图片补到同一 batch 大小；GT 和 proposals 继续用列表保存。
4. 每张图的 proposals 按 IoU 匹配 GT：`>=0.5` 是前景，`[0.1,0.5)` 是背景，更低的 IoU 暂不训练。从中抽样 RoI。
5. VGG16 对整张图片提取一次特征；RoI Pool 从特征图中得到每个 RoI 的 `7×7` 特征。
6. 分类头预测 21 类（20 个 VOC 类加背景）；框回归头预测每个前景类的 4 个偏移量。分类 loss 使用全部抽样 RoI，框 loss 只使用前景 RoI。
7. `backward()` 和 SGD 更新参数。单图推理时对预测框解码并逐类做 NMS。

## 目前的边界

- `predict.py` 是单图结果检查，不是 VOC mAP 评估。当前 proposal 生成脚本只处理 trainval；要测独立的 VOC 2007 test mAP，还需要为 test 生成 proposals，并实现正式的 AP/mAP 计算。
- 当前 `main.py` 在全部 trainval 上训练，没有单独验证集，因此只保存 `last.pth`，不产生 `best.pth`。不要用训练图片上的检测结果估计泛化精度。
- 框回归目标目前使用原始 `[dx, dy, dw, dh]`，尚未统计训练集均值/标准差进行标准化；这与论文的完整实验设置有差别。
- RoI Pool 现使用 torchvision 的同名算子，接口仍是 `[batch_idx, x1, y1, x2, y2]`。服务器建议使用 CUDA；CPU 可以运行，但完整训练会很慢。
