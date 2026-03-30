import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from types import SimpleNamespace

# ------------------------
# 1️⃣ 自定义数据集
# ------------------------
from Myself_train_model.datasets.cross_dataset_loader import CrossDatasetYOLO, collate_fn
from ultralytics.utils.loss import v8DetectionLoss  # 改好的 Loss

# ------------------------
# 2️⃣ 构造跨数据集
# ------------------------
dataset_list = [
    # Dataset 1: 只标注行人 (label 0)
    {
        'img_path': '/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_images_test/3c4d87ce4f898ae70d2d33b79e6c8f8a.jpeg',
        'bboxes': [[50, 60, 200, 220], [100, 120, 180, 200]],
        'labels': [0, 1],
        'ignore_class_ids': [2,3,4,5]  # 忽略车辆
    },
    # Dataset 2: 只标注车辆 (label 1)
    {
        'img_path': '/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_images_test/4a1a81f82ab76a6aa207222d1fe72629.jpg',
        'bboxes': [[30, 40, 120, 150],[60, 30, 80, 90]],
        'labels': [2,3],
        'ignore_class_ids': [0,1,4,5]  # 忽略行人
    },

    {
        'img_path': '/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_images_test/61d0639d0b2aa1043dba6a334dade8bb.jpeg',
        'bboxes': [[40, 40, 120, 150],[50, 30, 80, 90]],
        'labels': [4,5],
        'ignore_class_ids': [0,1,2,3]  # 忽略行人
    },
]

dataset = CrossDatasetYOLO(dataset_list, img_size=640)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# ------------------------
# 3️⃣ 加载模型
# ------------------------
model = YOLO("yolov8n.pt")  # 或 yolov8s.pt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.model.to(device)

# 设置默认超参
model.model.args = SimpleNamespace(
    box=7.5,
    cls=0.5,
    dfl=1.5,
)

# ------------------------
# 4️⃣ 初始化 Loss
# ------------------------
criterion = v8DetectionLoss(model.model)

# ------------------------
# 5️⃣ 优化器
# ------------------------
optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)

# ------------------------
# 6️⃣ 训练循环（跨数据集忽略未标注类别）
# ------------------------
epochs = 10

for epoch in range(epochs):
    model.model.train()
    total_loss = 0.0

    for imgs, gt_bboxes, gt_labels, mask_gt, ignore_class_ids in loader:
        imgs = imgs.to(device).float() / 255.0
        gt_bboxes = gt_bboxes.to(device).float()  # 🔹 确保 float, 不需要 grad
        gt_labels = gt_labels.to(device)

        # ------------------------
        # 构造 batch，忽略未标注类别
        # ------------------------
        batch_idx_list, cls_list, bboxes_list = [], [], []

        B, N = gt_labels.shape
        for b in range(B):
            ignore_ids = ignore_class_ids[b]
            for n in range(N):
                label = gt_labels[b, n].item()
                if label >= 0 and label not in ignore_ids:
                    batch_idx_list.append(b)
                    cls_list.append(label)
                    bboxes_list.append(gt_bboxes[b, n])

        if len(batch_idx_list) == 0:
            continue  # 本 batch 没有有效标注，跳过

        preds = model.model(imgs)

        batch = {
            "img": imgs,  # 🔥 必须有
            "batch_idx": torch.tensor(batch_idx_list, device=device),
            "cls": torch.tensor(cls_list, device=device),
            "bboxes": torch.stack(bboxes_list).to(device),
        }

        loss, loss_items = model.model.loss(batch, preds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ------------------------
        # 计算 loss
        # ------------------------
        # 并且 GT class 要映射到模型类别范围 [0, nc-1]
        # loss, _ = model.model.loss(batch, preds)  # 原生 loss 函数
        # loss, _ = model.model.loss(preds, batch)  # 原生 loss 函数
        # loss, _ = criterion(preds, batch)

        # loss = loss_dict['total']  # 或者 'box', 'cls', 'dfl' 等，根据版本

        # 🔥 测试用：构造一个简单 loss（一定有梯度）
        # loss = preds[1].sum()
        #
        # # ------------------------
        # # 反向
        # # ------------------------
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

    # ------------------------
    # 保存模型
    # ------------------------
    # torch.save(model.model.state_dict(), f"yolov8_cross_epoch_{epoch+1}.pt")

print("✅ Training Finished!")