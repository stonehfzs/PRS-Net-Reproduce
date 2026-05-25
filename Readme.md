# PRS-Net 复现运行说明

本项目是 PRS-Net 的 PyTorch 复现，用于无监督 3D 形状对称性检测。模型输入为 `32 x 32 x 32` voxel grid，输出：

- 3 个对称平面：`P_i = (n_x, n_y, n_z, d)`
- 3 个对称轴：`p_j = (w, x, y, z)`

## 环境

需准备 Python 3.10 版本并安装如下依赖：

```bash
pip install torch numpy scipy trimesh tqdm plotly
```

## 数据目录

原始 ModelNet 数据目录应按照如下形式保存：

```text
ModelNet40/
  airplane/
    train/
      airplane_0001.off
    test/
      airplane_0627.off
  chair/
    train/
    test/
```

## 预处理

普通预处理：

```bash
python preprocess.py --src ModelNet40 --dest ModelNet40_processed --workers 16
```

随机旋转增强训练集：

```bash
python preprocess.py --src ModelNet40 --dest ModelNet40_augmented_processed --workers 16 --augment-train --target-per-category 1000
```

## 训练

使用普通预处理数据训练：

```bash
python train.py --data-root ModelNet40_processed --split train --epochs 100 --batch-size 2048 --save-path prsnet_model.pth
```

使用增强数据训练：

```bash
python train.py --data-root ModelNet40_augmented_processed --split train --epochs 100 --batch-size 2048 --save-path prsnet_model.pth
```

常用参数：

```text
--data-root    预处理后的数据目录
--split        train / test / all
--epochs       训练轮数
--batch-size   batch size
--lr           学习率，默认1e-3
--w-r          正则项权重，默认25
--save-path    模型保存路径
```

正常训练建议使用：

```bash
--split train
```

## 单个样本可视化

```bash
python visualize.py --data ModelNet40_processed/airplane/test/airplane_0637.pt --model prsnet_model.pth --output visual_result.html
```

输出文件：

```text
visual_result.html
visual_result.log
```

`html` 文件用于交互式查看结果，`log` 文件保存误差、参数等细节信息。

## 批量可视化

每个类别的 test 前 20 个样本：

```bash
bash visualize_test20.sh ModelNet40_processed prsnet_model.pth visual_results_test20 20
```

使用增强预处理数据：

```bash
bash visualize_test20.sh ModelNet40_augmented_processed prsnet_model.pth visual_results_test20 20
```

## Rotation 诊断批处理

对几个明显可能存在旋转对称的类别进行批量验证：

```bash
bash diagnose_rotation.sh ModelNet40_augmented_processed prsnet_model.pth rotation_diagnostics_mixed 20
```

输出目录中会保存每个样本的：

```text
*.html
*.log
```

## 常见输出

训练结束后会生成：

```text
prsnet_model.pth
```

可视化或诊断结束后会生成：

```text
*.html
*.log
```
