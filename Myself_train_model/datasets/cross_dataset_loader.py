import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class CrossDatasetYOLO(Dataset):
    """
    跨数据集YOLO数据集，每张图片可以指定需要忽略的类别。
    自动返回 ignore_class_ids，方便训练时忽略未标注类别。
    """
    def __init__(self, dataset_list, img_size=640):
        """
        dataset_list: list of dict, 每个 dict 包含:
            - img_path: str
            - bboxes: list of [x1, y1, x2, y2]
            - labels: list of int
            - ignore_class_ids: list of int, 可选
        img_size: 输入模型的尺寸
        """
        self.dataset_list = dataset_list
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        data = self.dataset_list[idx]
        img = cv2.imread(data['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize + letterbox
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        nh, nw = int(h*scale), int(w*scale)
        img_resized = cv2.resize(img, (nw, nh))
        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        canvas[:nh, :nw, :] = img_resized
        img_tensor = torch.from_numpy(canvas).permute(2,0,1).float()  # C,H,W

        # bboxes -> tensor, xyxy
        bboxes = torch.tensor(data['bboxes'], dtype=torch.float32)
        labels = torch.tensor(data['labels'], dtype=torch.long)

        # ignore_class_ids
        ignore_class_ids = data.get('ignore_class_ids', [])

        # mask_gt 用于占位，可在后续Loss中使用
        mask_gt = torch.ones_like(labels, dtype=torch.bool)

        return img_tensor, bboxes, labels, mask_gt, ignore_class_ids


def collate_fn(batch):
    """
    批处理
    """
    imgs, bboxes, labels, masks, ignore_ids = zip(*batch)

    # pad bboxes/labels到最大数量
    max_num = max(b.shape[0] for b in bboxes)
    padded_bboxes, padded_labels, padded_masks = [], [], []

    for b, l, m in zip(bboxes, labels, masks):
        pad_num = max_num - b.shape[0]
        if pad_num > 0:
            b = torch.cat([b, torch.zeros((pad_num,4), dtype=torch.float32)], dim=0)
            l = torch.cat([l, -torch.ones(pad_num, dtype=torch.long)], dim=0)
            m = torch.cat([m, torch.zeros(pad_num, dtype=torch.bool)], dim=0)
        padded_bboxes.append(b)
        padded_labels.append(l)
        padded_masks.append(m)

    imgs = torch.stack(imgs)
    bboxes = torch.stack(padded_bboxes)
    labels = torch.stack(padded_labels)
    masks = torch.stack(padded_masks)

    return imgs, bboxes, labels, masks, ignore_ids