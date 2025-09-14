# DCGAN-PyTorch 项目说明

## 项目简介

本项目实现了基于 PyTorch 的深度卷积生成对抗网络（DCGAN），用于无监督学习和图像生成。代码结构清晰，适合 GAN/深度学习初学者学习和实验。项目代码基于[代码原作者仓库](https://github.com/gxwangupc/DCGAN-PyTorch)，进行修改。
数据源采用[70,171 张动漫头像图片](https://download.mindspore.cn/dataset/Faces/faces.zip)，图片大小均为 96\*96。

## 主要功能

- 支持人脸数据集（faces）训练和测试
- 训练过程自动保存模型和生成图片
- 损失曲线自动保存并可视化
- 固定随机种子，保证实验可复现
- 支持多进程数据加载加速训练

## 环境

- conda activate cartoon （本人的虚拟环境名字）
- Python 3.11.11
- PyTorch
- torchvision
- matplotlib
- 其他依赖见 environment.yml
- GPU 4060 laptop

## 目录结构

```
DCGAN-PyTorch/
├── main.py                 # 主训练脚本
├── test_main.py            # 测试脚本
├── Generator.py            # 生成器网络
├── Discriminator.py        # 判别器网络
├── plot_loss.py            # 损失曲线可视化
├── download_lsun.py        # LSUN数据集下载工具(原仓库数据集代码，本实验并不使用)
├── crop_from_grid.py       # 图片裁剪工
├── batch_crop_from_grid.py # 批量图片裁剪工具
├── faces/                  # 人脸图片数据集（需自行准备）
├── results/                # 训练结果（模型和图片）
├── .gitignore              # Git忽略配置
├── README.md               # 项目说明
└── environment.yml         # 环境配置文件
```

## 快速开始

### 1. 数据准备

将[动漫头像](https://download.mindspore.cn/dataset/Faces/faces.zip)按如下结构放置（每个类别一个子文件夹）：

```
faces/
└── faces/
    └── class1/
        ├── 0.jpg
        ├── 1.jpg
        └── ...
```

### 2. 测试环境

```bash
python test_main.py
```

- 仅用部分数据进行快速测试。

### 3. 训练模型

一个 epoch 大概 10 分钟。

```bash
python main.py
```

- 训练结果和模型会保存在 `./results` 目录。

### 4. 可视化损失曲线

```bash
python plot_loss.py
```

- 生成 `results/loss_curve.png`。

## 参数说明（main.py）

- `--dataroot` 数据集根目录
- `--batch_size` 批大小（默认 128）
- `--img_size` 输入图片尺寸（默认 64）
- `--nepoch` 训练轮数（默认 10）
- `--cuda` 使用 GPU 训练
- 其他参数见 main.py 注释

## 复现性说明

- 项目已固定随机种子（42），保证每次运行结果一致。
- 多进程数据加载不会影响训练结果。

## 备注

- faces.zip 等大文件未纳入版本控制，请自行准备数据集。
- 如需扩展到其他数据集或网络结构，可参考 main.py 代码结构进行修改。

---

如有问题或建议，欢迎 issue 或 PR！
