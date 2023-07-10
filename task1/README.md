# 实践I：基础实践
## 实践内容

- 学习目标：学习基础的网络结构和训练方法，了解深度学习中超参数的设置和搜索手段。
- 题目：CIFAR-10 Dataset 包含了 60k 张 32x32 的 RGB 彩色图片，这些图片共分为 10 个类别，
每个类别分别有 6k 张图片，其中 5k 作为该类别上的训练集、1k 作为测试集。使用 pytorch
框架实现 cifar-10 数据集上的图片分类任务，在训练集上进行训练，并在测试集上验

使用batch size=32，learning rate=0.001，SGD优化器。三层卷积网络：acc=63%

## 超参数研究
| Learning Rate | 0.01 | 0.001 | 0.0001 |
| ------------- | ---- | ----- | ------ |
| Accuracy      | -    | 0.63  | 0.68   |

| 优化器 | SGD(momentum=0.9) | Adam | Adagrad |
| ------ | ---- | ----- | ------ |
| Accuracy | 0.63  | 0.64  | 0.55   |

| Batch size | 16   | 32   | 64   | 128  |
| ---------- | ---- | ---- | ---- | ---- |
| Accuracy   | 0.62 | 0.63 | 0.65 | 0.64 |


## 网络优化
- 使用一个余弦退火学习率调度器 (CosineAnnealingLR)，初始学习率设置为0.01
- weight decay = 5e-4
- 使用特定参数对数据进行预处理： mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)

做出以上修改后，准确率可以在 10 个 epoch 内突破 80%，调整 ResNet 网络参数后，准确率可以达
到 85%。