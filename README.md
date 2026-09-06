# Segment Anything 人脸痘痘热力图分割

完整的 PyTorch + Segment Anything Model（SAM）语义分割项目。项目使用 SAM 图像编码器与可训练像素解码头，实现无需人工点/框提示的痘痘自动分割。

标签约定：

- `255`：痘痘区域，目标为 1；
- `0`：非痘痘区域，目标为 0；
- `128`：不确定区域，不参与 Loss 和评估。

数据路径、SAM 模型、Dataloader 类型、Loss、优化器、验证间隔和 Accelerator 参数全部由 YAML 控制。

## 为什么不直接用标签生成 SAM 提示框

原始 SAM 是提示式分割模型，需要点或框。若测试阶段从真实热力图生成提示，等于向模型泄露答案。本项目仅保留预训练 `vision_encoder`，增加自动二值分割头，因此 train、val、test 都只输入人像图，不读取标签来构造模型输入；标签只用于 Loss 和指标。

## 项目结构

```text
sam_acne_project/
├── sam_acne/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── loss.py
│   ├── metrics.py
│   ├── engine.py
│   └── utils.py
├── configs/
│   ├── train.yaml
│   └── smoke.yaml
├── tools/create_smoke_data.py
├── train.py
├── test.py
├── requirements.txt
└── pyproject.toml
```

## 数据格式

输入图片和标签按“相对路径 + 无扩展名文件名”配对，标签建议使用无损 PNG：

```text
dataset/train/
├── images/
│   ├── face_001.jpg
│   └── sub/face_002.jpeg
└── labels/
    ├── face_001.png
    └── sub/face_002.png
```

`strict_labels: true` 时，标签包含 0、128、255 以外的值会立即报错。标签缩放固定使用最近邻插值。

## 安装

```powershell
cd D:\sam_acne_project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

如使用 NVIDIA GPU，请先从 PyTorch 官网安装匹配本机 CUDA 的 PyTorch。首次正式运行会从 Hugging Face 下载 SAM 权重。

## 配置数据路径

编辑 `configs/train.yaml`：

```yaml
data:
  train:
    input_dir: D:/dataset/acne/train/images
    label_dir: D:/dataset/acne/train/labels
  val:
    input_dir: D:/dataset/acne/val/images
    label_dir: D:/dataset/acne/val/labels
  test:
    input_dir: D:/dataset/acne/test/images
    label_dir: D:/dataset/acne/test/labels
```

可从 YAML 切换组件：

```yaml
model:
  type: sam_encoder_segmenter
  pretrained_name: facebook/sam-vit-base
dataloader:
  type: standard              # standard | weighted
loss:
  type: masked_bce_dice       # masked_bce_dice | masked_focal_dice
```

SAM ViT-B 的绝对位置编码对应 1024×1024 输入，正式配置默认使用该尺寸。若要使用其他尺寸，需要更换支持位置编码插值的模型实现或重新训练位置编码，不能只修改 YAML 数字。

## 训练

```powershell
python train.py --config configs/train.yaml
```

多卡：

```powershell
accelerate config
accelerate launch train.py --config configs/train.yaml
```

- `train.val_every_steps`：每隔指定优化器 step 验证；
- `train.save_every_steps`：保存完整 Accelerator 状态；
- `train.resume_from`：从 `checkpoint-step-XXXXXX` 恢复；
- `model.freeze_encoder`：是否冻结 SAM 图像编码器；
- `optimizer.encoder_lr_multiplier`：编码器学习率相对解码头的倍率。

最佳权重默认保存到 `outputs/sam_acne/best/model.pt`。

## 测试

```powershell
python test.py --config configs/train.yaml
```

输出包括 Precision、Recall、Dice/F1、IoU、Accuracy、Loss，以及恢复到原图尺寸的 0—255 预测热力图。

## 完整 smoke 自检

无需下载 SAM 权重：

```powershell
python tools/create_smoke_data.py --output-dir smoke_data
python train.py --config configs/smoke.yaml
python test.py --config configs/smoke.yaml
```

`smoke.yaml` 的 `tiny_segmenter` 仅用于检查 Dataloader、128 ignore、Loss、反向传播、按 step 验证、checkpoint 和测试输出。正式训练请使用 `sam_encoder_segmenter`。

## 显存建议

SAM ViT-B 在 1024×1024 下显存占用较高。默认冻结 encoder、batch size 1，并开启梯度累积。需要全量微调时建议：

1. 将 `freeze_encoder` 改为 `false`；
2. 使用 fp16/bf16；
3. 保持 batch size 1 并增大 `gradient_accumulation_steps`；
4. 必要时启用梯度检查点；
5. 优先尝试 LoRA/Adapter，而不是直接缩小输入破坏 SAM 位置编码。

## 说明

项目只用于计算机视觉研究，不构成医疗诊断。人脸属于敏感数据，应取得授权，并按人物身份切分 train/val/test，避免身份泄漏。
