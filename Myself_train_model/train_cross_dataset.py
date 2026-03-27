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
    {
        'img_path': '/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_images_test/3c4d87ce4f898ae70d2d33b79e6c8f8a.jpeg',
        'bboxes': [[50, 60, 200, 220], [100, 120, 180, 200]],
        'labels': [0, 2]
    },
    {
        'img_path': '/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_images_test/4a1a81f82ab76a6aa207222d1fe72629.jpg',
        'bboxes': [[30, 40, 120, 150]],
        'labels': [1]
    },
]

dataset = CrossDatasetYOLO(dataset_list, img_size=640, ignore_class_ids=[])
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
# 6️⃣ 训练循环
# ------------------------
epochs = 50

for epoch in range(epochs):
    model.model.train()
    total_loss = 0.0

    for imgs, gt_bboxes, gt_labels, mask_gt in loader:
        imgs = imgs.to(device).float() / 255.0
        gt_bboxes = gt_bboxes.to(device)
        gt_labels = gt_labels.to(device)

        # ------------------------
        # 构造 batch
        # ------------------------
        batch_idx, cls, bboxes = [], [], []
        B, N = gt_labels.shape
        for b in range(B):
            for n in range(N):
                if gt_labels[b, n] >= 0:  # 🔥 忽略 -1
                    batch_idx.append(b)
                    cls.append(gt_labels[b, n])
                    bboxes.append(gt_bboxes[b, n])

        if len(batch_idx) == 0:
            continue

        batch = {
            "batch_idx": torch.tensor(batch_idx, device=device, dtype=torch.long),
            "cls": torch.tensor(cls, device=device, dtype=torch.long),
            "bboxes": torch.stack(bboxes).to(device),
        }

        # ------------------------
        # 前向
        # ------------------------
        preds = model.model(imgs)

        loss, _ = criterion(preds, batch)

        # ------------------------
        # 反向
        # ------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss:.4f}")

    # ------------------------
    # 保存模型
    # ------------------------
    torch.save(model.model.state_dict(), f"yolov8_cross_epoch_{epoch+1}.pt")

print("✅ Training Finished!")