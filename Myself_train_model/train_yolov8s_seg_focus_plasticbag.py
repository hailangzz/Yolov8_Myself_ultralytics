from ultralytics import YOLO

# # 训练地毯识别模型
# model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_wire.yaml")  # load a pretrained model (recommended for training)
# model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
# model.load("/home/chenkejing/PycharmProjects/ultralytics/runs/my_carpet_exp/yolov8_focus_v1/weights/last.pt")
# results = model.train(data="carpet_detect.yaml", epochs=100, imgsz=640, device=0, workers=0, resume=True, project="runs/my_carpet_exp", name="yolov8_focus_v1")
#


# # 训练线材检测模型
# model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8s_focus_carpet.yaml")  # load a pretrained model (recommended for training)
# # model.load("/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_carpet_exp/yolov8_focus_sa_v2/weights/best.pt")
# model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s.pt")
# results = model.train(data="carpet_detect.yaml", epochs=300, imgsz=640, device=-1, workers=0, batch=32, project="runs/my_carpet_exp", name="yolov8_focus_v")
#

"""
对应的命令行代码：

yolo segment train \
    model=/workspace/data/TrainingScript/wire_seg/yolov8-seg_focus_wire.yaml \
    data=/workspace/data/TrainingScript/wire_seg/seg_wire.yaml \
    pretrained=/workspace/runs/my_wire_seg_exp/yolov8s_wire_seg_v1_2/weights/last.pt \
    epochs=300 \
    imgsz=640 \
    batch=32 \
    workers=4 \
    amp=True \
    multi_scale=True \
    project=runs/my_wire_seg_exp \
    name=yolov8s_wire_seg_v1_ \
    augment=True \
    weight_decay=0.0005 \
    device=0
    
yolo segment train \
    model=/workspace/data/TrainingScript/wire_seg/yolov8-seg_focus_wire.yaml \
    data=/workspace/data/TrainingScript/wire_seg/seg_wire.yaml \
    epochs=300 \
    imgsz=640 \
    batch=32 \
    workers=4 \
    amp=True \
    project=runs/my_wire_seg_exp \
    name=yolov8s_wire_seg_v1_ \
    resume=True \
    augment=True \
    multi_scale=True \
    weight_decay=0.0005 \
    device=0
    
yolo segment train \
    model=/workspace/data/TrainingScript/wire_seg/yolov8-seg_focus_wire.yaml \
    data=/workspace/data/TrainingScript/wire_seg/seg_wire.yaml \
    epochs=300 \
    imgsz=640 \
    batch=32 \
    workers=4 \
    amp=True \
    project=runs/my_wire_seg_exp \
    name=yolov8s_wire_seg_v1_ \
    augment=True \
    multi_scale=True \
    weight_decay=0.0005 \
    device=0
    
    说明：
    amp=True 的作用：自动混合精度训练。带来的好处：显存减少 30~50%训练速度提升 20~60%可以用更大的 batch  
    workers=8 指的是 DataLoader 读取数据的并行进程数。CPU核心数推荐 8 推荐 workers=4; 核心数16 推荐workers=8
    augment=True :启动数据增强
    resume=True：自定加载，项目目录下的模型。与pretrained（手动指定预训练模型）一般不同时使用



# 线材小目标检测

tmux new -s yolov8_detect_training

yolo segment train \
    model=/workspace/data/TrainingScript/plasticbag_seg/yolov8-seg_focus_plasticbag_0330.yaml \
    data=/workspace/data/TrainingScript/plasticbag_seg/seg_plasticbag.yaml \
    epochs=300 \
    imgsz=960 \
    batch=36 \
    workers=4 \
    device=0 \
    amp=True \
    project=runs/my_plasticbag_seg_exp \
    name=yolov8s_plasticbag_seg_v1_ \
    save=True \
    save_period=15 \
    mosaic=0.5 \
    mixup=0.1 \
    close_mosaic=10 \
    weight_decay=0.0005

yolo segment train \
    model=/workspace/runs/my_plasticbag_seg_exp/yolov8s_plasticbag_seg_v1_2/weights/best.pt \
    data=/workspace/data/TrainingScript/plasticbag_seg/seg_plasticbag.yaml \
    epochs=300 \
    imgsz=960 \
    batch=36 \
    workers=4 \
    device=0 \
    amp=True \
    project=runs/my_plasticbag_seg_exp \
    name=yolov8s_plasticbag_seg_v1_ \
    save=True \
    save_period=15 \
    mosaic=0.5 \
    mixup=0.1 \
    close_mosaic=10
    
“”“
我给你按“**工程视角 + YOLO内部机制**”把这条命令拆开讲清楚，你看完基本就能知道每一行在干什么，以及为什么这样调。

---

# 🧠 一、整体作用一句话

这条命令是在做：

> 用自定义 YOLOv8 segmentation 模型，在 wire/cable 数据集上训练 300 epoch，并启用中等强度数据增强 + 定期保存模型。

---

# ⚙️ 二、逐参数解释（重点）

## 1️⃣ 模型与数据

### ```bash

model=/workspace/data/TrainingScript/wire_seg/yolov8-seg_focus_wire_0330.yaml

````

👉 作用：

- 用 **YOLO结构配置文件（yaml）定义网络**
- 不是加载权重，而是“从结构构建模型”

📌 意味着：

```text
从0初始化网络（随机权重）
````

如果你想微调 pretrained，应使用 `.pt`

---

### ```bash

data=/workspace/data/TrainingScript/wire_seg/seg_wire.yaml

````

👉 数据集配置：

包含：

- train/val路径
- class names
- nc类别数

例如：

```yaml
train: xxx/images/train
val: xxx/images/val
nc: 1
names: ["wire"]
````

---

# 🎯 2️⃣ 训练核心参数

### ```bash

epochs=300

````

训练 300 轮：

- epoch = 遍历一次完整训练集
- 15万样本的话，这个量是合理的

---

### ```bash
imgsz=960
````

👉 输入图像尺寸：

* 所有图片 resize 到 960×960
* YOLO segmentation 在这个尺寸更容易学细线结构

📌 影响：

| 越大  | 越小    |
| --- | ----- |
| 精度↑ | 速度↑   |
| 显存↑ | 模型泛化↑ |

---

### ```bash

batch=24

````

👉 每次更新用 24 张图

📌 影响：

- batch 越大 → 梯度更稳定
- batch 越小 → 泛化更强但训练噪声大

👉 你 5060 Ti + 960 分辨率：

```text
24 是比较合理的“卡边界值”
````

---

### ```bash

workers=8

````

👉 数据加载线程数

作用：

- 读图
- 解码
- 数据增强

📌 你之前 CPU 没跑满，这个参数其实可以再提高：

```text
8 → 12 或 16
````

---

### ```bash

device=0

````

👉 使用第 0 张 GPU

---

### ```bash
amp=True
````

👉 混合精度训练（FP16）

作用：

* 显存减半
* 速度更快
* 5060 Ti 必须开

---

# 💾 3️⃣ 保存机制（非常重要）

### ```bash

save=True

````

👉 开启模型保存

---

### ```bash
save_period=10
````

👉 每 10 epoch 保存一次模型

输出：

```text
weights/epoch10.pt
weights/epoch20.pt
...
```

📌 作用：

* 防止训练崩掉
* 可以回滚中间最优模型

---

# 🧠 4️⃣ 数据增强（关键）

## 🔥 mosaic=0.5

👉 Mosaic 拼图增强（YOLO核心增强）

作用：

* 4张图拼成1张
* 提高小目标能力
* 增强泛化

📌 0.5含义：

```text
50%概率使用mosaic
```

✔ 比 0.1 正常很多
✔ 对 wire/cable 很重要

---

## 🔥 mixup=0.1

👉 图像混合增强

```text
img1 + img2 → 叠加
```

作用：

* 防过拟合
* 提升鲁棒性

📌 0.1 = 轻微增强（合理）

---

## 🔥 close_mosaic=10

👉 训练后期关闭 mosaic

含义：

```text
最后10个epoch不使用 mosaic
```

📌 目的：

* 提高最终收敛精度
* 让模型“看真实分布”

---

# ⚖️ 5️⃣ 正则化（防过拟合）

### ```bash

weight_decay=0.0005

````

👉 L2正则

作用：

- 防止模型过拟合
- 控制权重大小

📌 YOLO默认值附近，合理

---

# 📦 6️⃣ loss权重

### ```bash
box=7.5
````

👉 bbox损失权重

📌 作用：

控制：

* 框回归学习强度

---

### ⚠️ 对 segmentation 的影响

YOLO segmentation loss结构：

```text
loss = box + cls + dfl + mask
```

你把：

```text
box=7.5（较高）
```

意味着：

👉 模型更关注“框准不准”
👉 mask学习相对被弱化一点

---

# 🧠 三、这套配置整体风格

你的配置属于：

## 👉 “中等增强 + 高分辨率 + 稳定收敛型训练”

特点：

### ✔ 优点

* 收敛稳定
* 不容易崩
* 适合工业数据（wire/cable）
* mask不会太飘

---

### ⚠️ 潜在问题

#### 1. mosaic 仍然偏低（0.5 → 可以0.7）

wire 是细长结构：

👉 mosaic太低 = 小目标泛化略弱

---

#### 2. batch 可能略大

24 @ 960：

👉 已经接近 GPU 饱和区

---

#### 3. box=7.5可能略压 mask

如果你目标是：

> 分割质量优先（而不是框）

可以调整：

```text
box=6.0
```

---

# 🚀 四、用一句话评价你的训练配置

👉 这是一个：

> “工程可用，但偏保守的 YOLO segmentation 训练配置”

---

# 🔥 五、如果你想进一步提升效果（关键建议）

如果你目标是 wire/cable segmentation（你这个场景很典型），我建议：

### ⭐ 提升优先级最高的3个改动：

#### 1️⃣ mosaic ↑

```bash
mosaic=0.7
```

---

#### 2️⃣ workers ↑（你CPU没用满）

```bash
workers=12~16
```

---

#### 3️⃣ batch 动态优化

如果显存允许：

```bash
batch=28 or 32
```

---

# 📌 如果你愿意下一步我可以帮你做

我可以帮你直接分析：

### 👉 “你这套YOLO segmentation到底是不是已经到瓶颈了”

包括：

* 当前GPU利用率是否合理
* loss曲线是否已经 plateau
* mask mAP为什么不涨
* 是否该换 YOLO11 / 改 head / 加 attention

甚至可以帮你判断：

> 👉 继续训练有没有意义，还是该改结构了

只要你把训练 log（最后50行）贴出来就行。

”“”
    
// 模型加载，断点续训（注：如果模型批次已经训练完成，就不会再进入训练）：
yolo segment train \
    data=/workspace/data/TrainingScript/wire_seg/seg_wire.yaml \
    model=/workspace/runs/my_wire_seg_exp/yolov8s_wire_seg_finetune_stage15/weights/last.pt \
    resume=True \
    epochs=90 \
    imgsz=960 \
    batch=36 \
    workers=5 \
    device=0 \
    amp=True \
    project=runs/my_wire_seg_exp \
    name=yolov8s_wire_seg_finetune_stage1 \
    save=True \
    save_period=10 \
    mosaic=0.5 \
    mixup=0.1 \
    close_mosaic=10 \
    weight_decay=0.0005 \



// 模型加载，继续训练（finetune、重训练）：
yolo segment train \
    data=/workspace/data/TrainingScript/wire_seg/seg_wire.yaml \
    model=/workspace/runs/my_wire_seg_exp/yolov8s_wire_seg_v2_6/weights/last.pt \
    epochs=100 \
    imgsz=960 \
    batch=36 \
    workers=5 \
    device=0 \
    amp=True \
    project=runs/my_wire_seg_exp \
    name=yolov8s_wire_seg_v2_ \
    save=True \
    save_period=15 \
    mosaic=0.5 \
    mixup=0.1 \
    close_mosaic=10 \
    weight_decay=0.0005 \
    box=7.5   
    
    
    
    说明：
    rect=True
    启用长宽比训练，避免原图 1902×1080 被强制缩放到 640×640 导致横向压缩。
    保持线材的形状比例，提高小目标检测能力。
    box=2.0
    mosaic=0 ：防止mosic图像增强时，破坏线材的特征连续性，且mosic后，线材过细，会直接被下采样消失掉。
    对 YOLOv8 来说，这个参数可以放大 box regression loss 的权重，对小目标更敏感。
    默认是 0.05~0.1 左右，你可以先试 2.0 或 1.5，看训练效果。

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    模型finetune两阶段
    注意：
        最优策略（强烈推荐）
        🥇 方法1：混合训练（最稳）
        real : public = 3:1 或 4:1 
          
        👉 核心思想：        
        用真实数据“拉方向”
        用公开数据“保基础”
        
    ✅ 给你改好的 finetune 版本（第一阶段）

        👉 适应真实数据（推荐先跑这个）
        
        yolo segment train \
            model=/workspace/runs/my_wire_seg_exp/yolov8s_wire_seg_v2_8/weights/last.pt \
            data=/workspace/data/TrainingScript/wire_seg/seg_wire_finetune.yaml \
            epochs=90 \
            imgsz=960 \
            batch=48 \
            workers=4 \
            amp=True \
            project=runs/my_wire_seg_exp \
            name=yolov8s_wire_seg_finetune_stage1 \
            augment=True \
            close_mosaic=10  \
            weight_decay=0.0005 \
            save_period=10 \
            device=0 \
            lr0=0.001 \
            freeze=10
        
        # 断点续训：
        yolo segment train \
          resume=True \
          project=runs/my_wire_seg_exp \
          name=yolov8s_wire_seg_finetune_stage15
          
        # 继续训练
        yolo segment train \
            model=/workspace/runs/my_wire_seg_exp/yolov8s_wire_seg_finetune_stage15/weights/best.pt \
            data=/workspace/data/TrainingScript/wire_seg/seg_wire_finetune.yaml \
            epochs=90 \
            imgsz=960 \
            batch=48 \
            workers=4 \
            amp=True \
            project=runs/my_wire_seg_exp \
            name=yolov8s_wire_seg_finetune_stage1 \
            augment=True \
            close_mosaic=10  \
            save_period=10 \
            weight_decay=0.0005 \
            device=0 \
            lr0=0.001 \
            freeze=10
            
        
              
            
    ✅ 第二阶段 finetune（解冻部分backbone层，当前冻结7层）
        
        👉 在第一阶段训练完之后，再跑：
        
        yolo segment train \
            model=/workspace/runs/my_wire_seg_exp/yolov8s_wire_seg_finetune_stage16/weights/best.pt \
            data=/workspace/data/TrainingScript/wire_seg/seg_wire_finetune.yaml \
            epochs=100 \
            imgsz=960 \
            batch=48 \
            workers=4 \
            amp=True \
            project=runs/my_wire_seg_exp \
            name=yolov8s_wire_seg_finetune_stage2 \
            augment=True \
            close_mosaic=10  \
            weight_decay=0.0003 \
            device=0 \
            freeze=7 \
            patience=25
            
            说明：patience=10 训练早停次数，如果val时10次都没有提升，则早停，防止过拟合
            
            参数说明：
                resume：
                
                情况1: 继续上一次训练（用 resume）
                                
                例如：                
                上次训练到了 epoch 60
                原计划训练 200 epoch
                因为断电、中断、显存问题停了
                
                这时候应该：
                
                yolo segment train \
                    resume=True \
                    model=/workspace/runs/my_wire_seg_exp/yolov8s_wire_seg_v2_6/weights/last.pt
                
                或者：
                
                yolo resume \
                    model=/workspace/runs/my_wire_seg_exp/yolov8s_wire_seg_v2_6/weights/last.pt
                
                恢复内容包括：
                
                模型权重
                Optimizer状态
                LR Scheduler状态
                EMA
                当前epoch
                
                相当于：
                
                从断点继续跑
                
                情况2：重新开始一轮 Finetune（你的情况更像这个）
                
                例如：
                
                已有一个训练好的模型
                新增了一批数据
                想重新设定学习率
                想修改 freeze
                想修改增强策略
                
                这时候不要用 resume。
                
                直接：
                
                yolo segment train \
                    model=/workspace/runs/my_wire_seg_exp/yolov8s_wire_seg_v2_6/weights/last.pt \
                    data=seg_wire_finetune.yaml \
                    epochs=90 \
                    lr0=0.001 \
                    freeze=10
                
                这样：
                
                权重从 last.pt 加载
                Optimizer重新初始化
                学习率重新从 0.001 开始
                训练记录重新开始
                
                这才是真正意义上的：
                
                Fine-tune
"""

"""
from ultralytics import YOLO

if __name__ == "__main__":

    # 1️⃣ 加载分割模型结构（seg）
    model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8-seg_focus_wire.yaml")

    # 2️⃣ 加载预训练权重（非常重要）
    model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s-seg.pt")
    # model.load("/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_wire_seg_exp/yolov8s_wire_seg_v1_2/weights/last.pt")

    # 3️⃣ 开始训练
    results = model.train(
        task="segment",                 # ⭐ 必须指定
        data="coco8-seg_wire.yaml",      # 分割数据集 yaml
        epochs=300,
        imgsz=640,
        batch=12,                       # seg 比 detect 更吃显存
        device=0,                       # -1 = CPU，0 = GPU
        workers=4,
        optimizer="SGD",                # ⭐ 稳定，利于部署
        lr0=0.001,
        patience=20,
        project="runs/my_wire_seg_exp",
        name="yolov8s_wire_seg_v1_",
        resume=True
    )
"""


if __name__ == "__main__":
    # 1️⃣ 加载分割模型结构（seg）
    model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/ultralytics/cfg/models/v8/yolov8-seg_focus_wire.yaml")

    # 2️⃣ 加载预训练权重（非常重要）
    model.load("/home/chenkejing/PycharmProjects/ultralytics/yolov8s-seg.pt")
    # model.load("/home/chenkejing/PycharmProjects/ultralytics/Myself_train_model/runs/my_wire_seg_exp/yolov8s_wire_seg_v1_2/weights/last.pt")

    # 3️⃣ 开始训练
    results = model.train(
        task="segment",  # ⭐ 必须指定
        data="coco8-seg_wire.yaml",  # 分割数据集 yaml
        epochs=300,
        imgsz=640,
        batch=12,  # seg 比 detect 更吃显存
        device=0,  # -1 = CPU，0 = GPU
        workers=4,
        optimizer="SGD",  # ⭐ 稳定，利于部署
        cos_lr=True,
        amp=False,  # ⭐ ONNX/RKNN 强烈建议关  /amp 控制的是「训得快不快 vs 稳不稳」
        augment=True,
        dropout=0.1,
        project="runs/my_wire_seg_exp",
        name="yolov8s_wire_seg_v1_",
        # resume=False                    # 控制的是「训不训旧的状态」
        resume=True,
    )

    # watch -n 1 nvidia-smi #监控GPU占用信息
