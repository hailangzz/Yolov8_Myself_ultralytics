import torch
from torch.utils.data import DataLoader
from Myself_train_model.datasets.cross_dataset_loader import CrossDatasetYOLO, collate_fn

# 假设你已经有TaskAlignedAssigner类
from ultralytics.utils.tal import TaskAlignedAssigner  # 替换成你的assigner路径

# ------------------------
# 构造跨数据集
# ------------------------
dataset_list = [
    {'img_path':'/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_images_test/2d961f6d13e98a4b.jpg',
     'bboxes':[[50,60,200,220],[100,120,180,200]],
     'labels':[0,2]},
    {'img_path':'/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_images_test/3DGenerate_000001.png',
     'bboxes':[[30,40,120,150]],
     'labels':[1]},
    # 可以继续添加其他数据集图片
]

ignore_classes = []  # 如果需要忽略某些类别填id，例如[-1]表示ignore

dataset = CrossDatasetYOLO(dataset_list, img_size=640, ignore_class_ids=ignore_classes)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# ------------------------
# 初始化 TaskAlignedAssigner
# ------------------------
num_classes = 5  # 根据你的任务改
assigner = TaskAlignedAssigner(topk=13, num_classes=num_classes)

# 假设你已经有预测的 pd_scores, pd_bboxes, anc_points
for imgs, gt_bboxes, gt_labels, mask_gt in loader:
    # 这里 imgs -> [B,C,H,W]
    # gt_bboxes -> [B, n_max_boxes,4]
    # gt_labels -> [B, n_max_boxes] (-1为ignore)
    # mask_gt -> [B, n_max_boxes,1]

    # 假设 pd_scores, pd_bboxes, anc_points 是模型输出
    # target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = \
    #     assigner(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

    print('imgs', imgs.shape)
    print('gt_labels', gt_labels)
    print('mask_gt', mask_gt)
    break
