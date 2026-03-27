import torch
from torch.utils.data import Dataset
import cv2
import os

class CrossDatasetYOLO(Dataset):
    """
    跨数据集 YOLOv8 数据集
    每个样本返回:
        image: tensor [C,H,W]
        gt_bboxes: tensor [n_max_boxes,4] xyxy
        gt_labels: tensor [n_max_boxes] (-1表示ignore)
        mask_gt: tensor [n_max_boxes,1] (有效gt标记)
    """
    def __init__(self, dataset_list, img_size=640, ignore_class_ids=None):
        """
        Args:
            dataset_list: list of dict，每个dict包含 'img_path', 'bboxes', 'labels'
            img_size: 图片resize大小
            ignore_class_ids: list[int] 忽略类别id
        """
        self.dataset_list = dataset_list
        self.img_size = img_size
        self.ignore_class_ids = ignore_class_ids if ignore_class_ids else []

        # 计算 n_max_boxes
        self.n_max_boxes = max([len(d['labels']) for d in dataset_list])

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        data = self.dataset_list[idx]
        img_path = data['img_path']
        bboxes = data['bboxes']  # list of [x_min,y_min,x_max,y_max]
        labels = data['labels']  # list of int

        # 读取图片
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0  # [C,H,W]

        # pad gt_bboxes 和 gt_labels
        gt_bboxes = torch.zeros((self.n_max_boxes,4), dtype=torch.float32)
        gt_labels = torch.full((self.n_max_boxes,), -1, dtype=torch.int64)  # 默认-1 ignore
        mask_gt = torch.zeros((self.n_max_boxes,1), dtype=torch.float32)

        n = len(labels)
        if n>0:
            gt_bboxes[:n] = torch.tensor(bboxes, dtype=torch.float32)
            gt_labels[:n] = torch.tensor(labels, dtype=torch.int64)
            mask_gt[:n] = 1.0

            # 处理 ignore_class_ids
            for i, lbl in enumerate(labels):
                if lbl in self.ignore_class_ids:
                    gt_labels[i] = -1  # 标记ignore

        return img, gt_bboxes, gt_labels, mask_gt

def collate_fn(batch):
    """Dataloader collate_fn"""
    imgs, gt_bboxes, gt_labels, mask_gt = zip(*batch)
    imgs = torch.stack(imgs)
    gt_bboxes = torch.stack(gt_bboxes)
    gt_labels = torch.stack(gt_labels)
    mask_gt = torch.stack(mask_gt)
    return imgs, gt_bboxes, gt_labels, mask_gt