# VAND 4.0 INP Residual

这是一个面向 MVTec AD 2 / VAND 4.0 的异常分割训练、验证和提交生成工程。当前版本以 INP-Former 的重建异常图为主干，并加入残差特征头、CDO loss 和 synthetic anomaly 训练。

## 当前指标口径

本项目现在有两套 SegF1 口径：

1. `train.py` 的周期验证默认使用 `SegF1Max`。
   每 `train.eval_interval` 个 epoch 在 `test_public` 上验证一次，并从带标注的 public mask 中寻找最优阈值，用来观察当前模型的 SegF1 上限。

2. `evaluate.py` 和 `test.py` 使用 validation 阈值。
   阈值从各类别 `validation/good` 正常样本的 anomaly map 像素分位数估计，默认分位数是 `0.9995`。这条路径更接近真实提交时不能看测试标签的使用方式。

如果希望训练阶段也使用 validation 阈值口径，把配置里的：

```yaml
train:
  eval_metric: segf1
```

改成 `segf1` 即可。默认是 `segf1max`。

## 数据目录

默认数据根目录在 `configs/mvtecad2_inp_residual.yaml` 中配置：

```yaml
data:
  root: /home/Groups/group1/Working/ljx/Dataset/defect_dataset/mvtec_ad_2
```

每个类别目录需要符合下面的结构：

```text
mvtec_ad_2/
  can/
    train/good/
    validation/good/
    test_public/good/
    test_public/bad/
    test_public/ground_truth/bad/
    test_private/
    test_private_mixed/
  fabric/
  ...
```

当前默认类别：

```text
can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts
```

也可以通过命令行覆盖类别，例如：

```bash
--categories can,fabric
```

## 环境安装

先按机器 CUDA 版本安装 PyTorch，然后安装项目依赖：

```bash
cd /home/Groups/group1/Working/ljx/VAND4/vand4.0_baseline_1
pip install -r requirements.txt
```

默认配置会读取：

```yaml
model:
  backbone:
    source_root: third_party/INP-Former-main
    weights_dir: backbones/weights
```

项目内已包含 INP-Former 源码和 DINOv2 权重，上传 `vand4.0_baseline_1_copy` 时不再依赖同级目录里的外部工程。

## 训练

推荐显式指定 GPU，避免脚本默认 GPU 和当前机器环境不一致：

```bash
python train.py \
  --config configs/mvtecad2_inp_residual.yaml \
  --data_root /home/Groups/group1/Working/ljx/Dataset/defect_dataset/mvtec_ad_2 \
  --save_dir checkpoints/vand4_large \
  --gpu_id 0
```

也可以使用脚本：

```bash
GPU_ID=0 bash scripts/train_mvtecad2.sh
```

训练过程会保存：

```text
checkpoints/vand4/latest.pth
checkpoints/vand4/epoch_010.pth
checkpoints/vand4/epoch_020.pth
...
```

注意：当前 `train.py` 不会自动生成 `best.pth`。如果后续测试命令写的是 `best.pth`，需要先确认该文件真实存在，或者改成 `latest.pth` / `epoch_XXX.pth`。

断点恢复：

```bash
python train.py \
  --config configs/mvtecad2_inp_residual.yaml \
  --data_root /home/Groups/group1/Working/ljx/Dataset/defect_dataset/mvtec_ad_2 \
  --save_dir checkpoints/vand4 \
  --resume checkpoints/vand4/latest.pth \
  --gpu_id 0
```

训练阶段关键配置：

```yaml
train:
  epochs: 200
  batch_size: 4
  save_interval: 10
  eval_public: true
  eval_interval: 10
  eval_metric: segf1max
  eval_threshold_quantile: 0.9995
  eval_gaussian_sigma: 4.0
```

## 评估

`evaluate.py` 默认评估 `test_public`，并使用 `validation/good` 估计出来的阈值计算 `Pixel SegF1`：

```bash
python evaluate.py \
  --config configs/mvtecad2_inp_residual.yaml \
  --checkpoint checkpoints/vand4/epoch_110.pth \
  --data_root /home/Groups/group1/Working/ljx/Dataset/defect_dataset/mvtec_ad_2 \
  --gpu_id 0
```

常用覆盖参数：

```bash
python evaluate.py \
  --config configs/mvtecad2_inp_residual.yaml \
  --checkpoint checkpoints/vand4/epoch_110.pth \
  --categories can,fabric \
  --splits test_public \
  --threshold_quantile 0.9995 \
  --gpu_id 0
```

输出中会包含：

```text
Public SegF1 threshold@...: ...
can Public Pixel Auroc:..., Pixel SegF1:..., threshold:...
Public Mean Pixel Auroc:..., Pixel SegF1:...
```

这里的 `SegF1` 不是 `SegF1Max`，而是 validation 阈值下的 SegF1。

## 生成测试结果

生成官方提交用连续 anomaly map：

```bash
python test.py \
  --config configs/mvtecad2_inp_residual.yaml \
  --checkpoint checkpoints/vand4/latest.pth \
  --data_root /home/Groups/group1/Working/ljx/Dataset/defect_dataset/mvtec_ad_2 \
  --output_dir outputs/vand4_submit \
  --save_tiff \
  --no_save_thresholded \
  --gpu_id 0
```

同时生成二值 mask：

```bash
python test.py \
  --config configs/mvtecad2_inp_residual.yaml \
  --checkpoint checkpoints/vand4/latest.pth \
  --data_root /home/Groups/group1/Working/ljx/Dataset/defect_dataset/mvtec_ad_2 \
  --output_dir outputs/vand4_submit \
  --save_tiff \
  --save_thresholded \
  --gpu_id 0
```

当启用 `--save_thresholded` 且没有手动传 `--threshold` 时，`test.py` 会从 `validation/good` 样本估计阈值：

```text
thresholded mask validation threshold@...: ...
```

如果要固定阈值：

```bash
python test.py \
  --config configs/mvtecad2_inp_residual.yaml \
  --checkpoint checkpoints/vand4/latest.pth \
  --output_dir outputs/vand4_submit \
  --save_tiff \
  --save_thresholded \
  --threshold 0.2 \
  --gpu_id 0
```

默认测试 split 来自配置：

```yaml
test:
  splits:
    - test_private
    - test_private_mixed
```

输出目录结构：

```text
outputs/vand4_submit/
  anomaly_images/
    can/test_private/*.tiff
    can/test_private_mixed/*.tiff
  anomaly_images_thresholded/
    can/test_private/*.png
    can/test_private_mixed/*.png
  analysis_panels/
    can/test_private/*.png
    can/test_private_mixed/*.png
```

说明：`analysis_panels` 中每张图为横向拼接的原图、二值阈值图和热力图，三个子图都会缩放到 `392x392`。

## 常见问题

`data_root does not exist`

检查 `configs/mvtecad2_inp_residual.yaml` 里的 `data.root`，或者在命令行传 `--data_root` 覆盖。

`--gpu_id was set, but CUDA is not available`

当前环境没有可用 CUDA。可以改用 `--device cpu` 做小规模调试，正式训练建议使用 CUDA。

`checkpoint not found`

训练脚本只自动保存 `latest.pth` 和 `epoch_XXX.pth`。测试或评估时不要直接使用不存在的 `best.pth`。

`No normal validation scores found`

`evaluate.py` / `test.py --save_thresholded` 需要每个类别存在 `validation/good` 正常样本，用来估计阈值。
