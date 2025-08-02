# CycleGAN-Ghibli 风格迁移

项目使用 CycleGAN 实现真实图片到吉卜力风格的图像转换。使用改进的 UNet++ 作为生成器，PatchGAN 作为判别器，实现了高质量的风格迁移。
备选的生成器模型包括 UNet、ResNet 和 Transformer，备选的判别器模型包括 PatchGAN 和 Multi-Scale Edge，用于消融实验。

## 项目链接
- GitHub: https://github.com/cyxCHENYUXUAN/CycleGAN-Ghibli

## 项目结构

```
CycleGAN-Ghibli/
├── models/
│   ├── Discriminator/
│   │   ├── PatchGAN70Discriminator.py  # 判别器模型
│   │   └── MultiScaleEdgeDiscriminator.py
│   └── Generator/
│       ├── UNetPPGenerator.py          # 主要使用的生成器模型
│       ├── UNetGenerator.py
│       ├── ResNetGenerator.py
│       └── TransformerGenerator.py
├── utils/
│   ├── dataset.py                    # 数据集加载
│   ├── loss.py                       # 损失函数实现
|   ├── metrics.py                    # 评价指标（尚未实现）
|   ├── prepare_test_dataset.py       # 保存生成的图片
│   ├── ReplayBuffer.py               # 经验回放缓冲区
│   └── utils.py                      # 工具函数
├── Data/
│   └── dataset/
│       ├── trainA/         # 真实图片训练集
│       └── trainB_ghibli/  # 吉卜力风格图片训练集
├── train.py                # 训练脚本
├── test_env.py             # 环境测试脚本
└── requirements.txt        # 项目依赖
```

## 环境配置

1. 创建并激活虚拟环境（推荐）：
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 验证环境：
```bash
python test_env.py
```

## 数据集准备

项目使用 [Real to Ghibli Image Dataset](https://www.kaggle.com/datasets/shubham1921/real-to-ghibli-image-dataset-5k-paired-images)，包含5000对真实图片和吉卜力风格图片。

1. 从 Kaggle 下载数据集
2. 在项目根目录下创建 `Data` 目录，并将下载的数据集解压到该目录下

3. 验证数据集：
```bash
cd utlis
python dataset.py
```
如果一切正常，会显示一对真实图片和吉卜力风格图片的对比。（需取消注释 `dataset.py` 中的 `if __name__ == '__main__` 函数）

## 训练模型

```bash
python train.py
```

- 模型检查点会保存在 `checkpoints` 目录
- 生成的示例图片会保存在 `training_examples` 目录

## 当前模型架构

- **生成器**：改进的 UNet++ 架构，具有更好的特征提取和重建能力
- **判别器**：70x70 PatchGAN，用于判别局部图像块的真实性
- **损失函数**：
  - 对抗损失（MSE）
  - 循环一致性损失（L1）
  - 身份损失（L1）

## TODO

- [ ] 使用不同架构的生成器与判别器进行实验
- [ ] 实现评价指标
- [ ] 添加推理脚本
- [ ] 优化训练速度
- [ ] 添加更多的数据增强方法
- [ ] 支持多GPU训练
- [ ] 完成前端展示页面
