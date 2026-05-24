# PRS-Net 复现

本项目是 PRS-Net 的 PyTorch 复现，用于无监督 3D 形状对称性检测。模型输入为 `32 x 32 x 32` voxel grid，输出：

- 3 个候选反射平面：`P_i = (n_x, n_y, n_z, d)`
- 3 个候选旋转参数：单位四元数 `p_j = (w, x, y, z)`

训练目标包含：

- `L_sd`：对称距离损失
- `L_r`：正则项，用于避免多个 plane / axis 重复
- `L = L_sd + w_r * L_r`

## 文件说明

```text
preprocess.py        将 .off mesh 预处理为 .pt 张量
dataset.py           从预处理好的 .pt 文件中读取训练数据
model.py             PRS-Net 3D CNN 模型
loss.py              对称距离损失和正则损失
train.py             训练脚本
validation.py        基于 SDE 的候选结果筛选
visualize.py         单个样本的预测和可视化
visualize_test20.sh  每个 category 的 test 前 20 个样本批量可视化
```

## 数据目录结构

原始数据使用 ModelNet 风格目录：

```text
ModelNet40/
  airplane/
    train/
      airplane_0001.off
      ...
    test/
      airplane_0627.off
      ...
  chair/
    train/
    test/
  ...
```

预处理后会保留相同的 category / split 结构：

```text
ModelNet40_processed/
  airplane/
    train/
      airplane_0001.pt
    test/
      airplane_0627.pt
  ...
```

## 预处理

标准预处理：

```bash
python preprocess.py --src ModelNet40 --dest ModelNet40_processed --workers 16
```

如果不指定 `--dest`，输出目录会根据 `--src` 自动生成：

```bash
python preprocess.py --src ModelNet40
```

输出为：

```text
ModelNet40_processed
```

每个 `.pt` 文件包含：

```text
voxels          (L, L, L)       表面 occupancy grid，作为网络输入
surface_points  (1000, 3)       表面采样点 Q = {q_k}，用于 L_sd
closest_grid    (L, L, L, 3)    近似最近表面点查表
source_off       str            原始 .off 文件路径
rotation_seed    int 或 None     随机增强种子
```

默认 voxel 分辨率：

```text
L = 32
```

## 随机旋转增强

为了接近论文中的随机旋转增强设置，可以使用：

```bash
python preprocess.py --src ModelNet40 --workers 16 --augment-train --target-per-category 1000
```

如果不指定 `--dest`，增强数据会写入单独目录：

```text
ModelNet40_augmented_processed
```

增强模式下：

- `train` split 会有放回随机采样
- 每个生成的 train 样本会应用一次随机 SO(3) 旋转
- 每个 category 生成 `--target-per-category` 个训练样本
- `test` split 不做增强，只正常预处理一次

例如每类生成 4000 个训练样本：

```bash
python preprocess.py --src ModelNet40 --workers 16 --augment-train --target-per-category 4000
```

## 训练

使用标准预处理数据训练：

```bash
python train.py --data-root ModelNet40_processed --split train --epochs 100 --batch-size 32
```

使用随机增强数据训练：

```bash
python train.py --data-root ModelNet40_augmented_processed --split train --epochs 30 --batch-size 32
```

常用参数：

```text
--data-root   预处理后的数据根目录
--split       train / test / all
--epochs      训练 epoch 数
--batch-size  batch size
--lr          Adam 学习率，默认 1e-3
--w-r         正则损失权重，默认 25
--save-path   模型保存路径，默认 prsnet_model.pth
```

正常训练应使用：

```text
--split train
```

不要在正常训练中使用 `--split all`，否则会把 test 数据混入训练。

## 验证规则

`validation.py` 使用以下规则筛选模型输出：

```text
1. SDE error <= threshold
2. plane normal 之间夹角 > 30 degrees
3. rotation axis 必须通过每 1 degree 的旋转测试
4. rotation axis 之间夹角 > 30 degrees
```

默认阈值：

```text
threshold = 4e-4
```

当前 SDE 使用归一化平方距离：

```text
E(S, tau) = (1 / |S|) * sum(dist(tau(p), S)^2)
```

其中 `S` 由 occupied voxel coordinates 表示。

## 单样本可视化

运行：

```bash
python visualize.py --data ModelNet40_processed/airplane/test/airplane_0637.pt --model prsnet_model.pth
```

终端会输出：

- validated plane / axis 数量
- 每个候选 reflection plane 的误差
- 每个候选 reflection plane 的参数
- 每个候选 rotation axis 的误差
- 每个候选 rotation axis 的 quaternion、axis 和 angle

单样本模式下，`visualize.py` 默认写出：

```text
visual_result.html
```

也可以指定输出路径：

```bash
python visualize.py --data ModelNet40_processed/airplane/test/airplane_0637.pt --model prsnet_model.pth --output airplane_0637.html
```

HTML 中只绘制通过 validation 的 plane 和 axis。未通过的候选会打印在终端里，但不会画到图中。

## 批量可视化

每个 category 的 test 前 20 个样本批量可视化：

```bash
bash visualize_test20.sh ModelNet40_processed prsnet_model.pth visual_results_test20
```

参数含义：

```text
第 1 个参数：processed data root
第 2 个参数：model path
第 3 个参数：output directory
```

输出结构：

```text
visual_results_test20/
  airplane/
    airplane_0627.html
    ...
  chair/
    ...
```

批处理时，脚本会通过 `--output` 让 `visualize.py` 直接写入目标文件，例如：

```text
visual_results_test20/airplane/airplane_0627.html
```

## 注意事项

- 当前使用的是 ModelNet40，不完全等同于原论文中的 ShapeNet 设置。
- 原论文使用了大量随机旋转增强；如果只用原始 ModelNet40，结果会受到轴对齐偏置影响。
- `--target-per-category 1000` 对 ModelNet40 大约会生成 40,000 个训练样本。
- 增强数量越大，覆盖越充分，但预处理文件体积和训练时间也会增加。
- 当前训练 loss 在 batch 内逐样本计算，因此训练时间会随预处理样本数量近似线性增加。
