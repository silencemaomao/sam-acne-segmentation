# Segment Anything 互斥多类别人脸瑕疵分割

完整的 PyTorch、SAM 图像编码器与 Accelerate 项目，支持类别互斥、0/128/255 热力图，
以及同一数据集内混合的部分/完整标注。模型仅从人像自动预测，不会用真实 mask 构造
SAM 点或框，避免测试阶段标签泄漏。

## 标签语义与串扰处理

模型输出“背景 + data.classes”共 C+1 个 logits，并使用 softmax，从结构上保证一个像素
只预测一个类别。每张已有类别 mask 会约束该像素的允许类别集合：

| 当前类别 mask | 允许类别 | 含义 |
|---|---|---|
| 255 | 仅当前类别 | 确定属于当前类，并排除背景及其他类 |
| 128 | 当前类别或背景 | 当前类不确定，但确定不属于其他瑕疵类 |
| 0 | 除当前类别外的所有类 | 仅确定不属于当前类 |
| mask 缺失 | 不增加约束 | 未标注，绝不自动当作负样本 |

多个 mask 的约束取交集。只有所有类别均有 mask 且某像素全为 0 时，才能确定它是背景；
若只有 acne mask 且值为 0，该像素仍可属于背景或任一未标注类别。两个类别 255 重叠，
或一个类别 128 与另一类别 255 重叠，会被 Dataset 识别为互斥冲突并报告位置。

partial_label_ce_dice 使用集合标签交叉熵：

~~~text
-log(sum(softmax(logits)[允许类别]))
~~~

Dice 与验证指标仅在标注将类别唯一确定的像素上计算，避免未知类别制造错误负样本。

## 数据格式

图片和 mask 按相对路径（忽略扩展名）配对。类别目录或单个 mask 可以缺失：

~~~text
dataset/train/
├── images/
│   ├── face_001.jpg
│   ├── face_002.jpg
│   └── face_003.jpg
└── labels/
    ├── acne/
    │   ├── face_001.png      # 只标 acne
    │   └── face_003.png      # face_003 全标
    ├── pigmentation/
    │   ├── face_002.png      # 只标 pigmentation
    │   └── face_003.png
    └── scar/
        └── face_003.png
~~~

strict_labels: true 会拒绝 0、128、255 以外的值；mask 应使用无损 PNG，缩放固定为最近邻。
每张图默认至少需要一个 mask。完全无标签图片可通过下列配置跳过：

~~~yaml
data:
  unlabeled_image_policy: skip   # error | skip
~~~

## 安装与配置

~~~powershell
cd D:\sam_acne_project
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
~~~

正式配置见 configs/train.yaml，所有路径和组件均由 YAML 控制：

~~~yaml
data:
  classes: [acne, pigmentation, scar]
model:
  type: sam_encoder_segmenter
  pretrained_name: facebook/sam-vit-base
  num_classes: null  # 自动为背景 + 3 类
loss:
  type: partial_label_ce_dice
  ce_weight: 1.0
  dice_weight: 1.0
  class_weights: {background: 1.0, acne: 4.0, pigmentation: 2.0, scar: 3.0}
~~~

SAM ViT-B 的绝对位置编码对应 1024×1024，正式配置默认使用该尺寸。项目保留预训练
vision encoder 并增加 prompt-free 像素解码头。

注意：v0.3 起输出通道由 C 改为 C+1，旧版独立 sigmoid 模型的 checkpoint 结构不兼容，
需按新配置重新训练。

## 训练与测试

~~~powershell
python train.py --config configs/train.yaml
python test.py --config configs/train.yaml
~~~

多卡训练：

~~~powershell
accelerate config
accelerate launch train.py --config configs/train.yaml
~~~

train.val_every_steps 与 save_every_steps 按优化器 step 计数，train.resume_from 可恢复完整
Accelerator 状态。测试使用 softmax 概率，只为瑕疵类别保存
“输出目录/类别/相对文件名.png”，不会导出背景通道。

## 离线完整 smoke test

无需下载 SAM 权重：

~~~powershell
python tools/create_smoke_data.py --output-dir smoke_data_partial
python train.py --config configs/smoke.yaml
python test.py --config configs/smoke.yaml
~~~

生成数据混合了单类别部分标注与全类别标注，用来验证 Dataloader、集合 Loss、反向传播、
按 step 验证、checkpoint 和分类别热力图导出。正式训练请使用 sam_encoder_segmenter。

## 说明

项目只用于计算机视觉研究，不构成医疗诊断。人脸属于敏感数据，应取得授权并按人物身份
切分 train/val/test，避免身份泄漏。
