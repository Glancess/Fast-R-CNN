from torchvision.datasets import VOCDetection

voc = VOCDetection(
    root="./data",
    year="2007",
    image_set="trainval",
    download=True,
)

print("dataset size:", len(voc))

image, target = voc[0]

print(image)
print(image.size)
print(target)
