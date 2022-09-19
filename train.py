# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 15:07
@Author        : Tianxiaomo
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import time
import logging
import os, sys, math
import argparse
from collections import deque
import datetime

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict

from dataset import Yolo_dataset
from cfg import Cfg
from models import Yolov4
from tool.darknet2pytorch import Darknet

from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])  # (N, K, 2)
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])  # (N, K, 2)
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared, 中心点距离的平方
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)  # (N, )
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)  # (K, )
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)  # (N, K)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())  # (N, K)
    area_u = area_a[:, None] + area_b - area_i  # (N, K)
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32)
            # i = 0  masked_anchors:  输出步幅为8时（输出尺寸76x76），先验框在输出特征图中的尺寸
            # array([[1.5, 2.],
            #        [2.375, 4.5],
            #        [5., 3.5]], dtype=float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # 输出步幅为8时（输出尺寸76x76），所有先验框在输出特征图中的尺寸（右边两列）
            # array([[0., 0., 1.5, 2.],
            #        [0., 0., 2.375, 4.5],
            #        [0., 0., 5., 3.5],
            #        [0., 0., 4.5, 9.375],
            #        [0., 0., 9.5, 6.875],
            #        [0., 0., 9., 18.25],
            #        [0., 0., 17.75, 13.75],
            #        [0., 0., 24., 30.375],
            #        [0., 0., 57.375, 50.125]], dtype=float32)

            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]  # i=0 76
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)
            # grid_x[0, 0]             grid_x shape [batch, 3, 76, 76]
            # tensor([[0., 1., 2., ..., 75.],
            #         [0., 1., 2., ..., 75.],
            #         [0., 1., 2., ..., 75.],
            #         [0., 1., 2., ..., 75.],
            #         [0., 1., 2., ..., 75.]])
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)
            # grid_y[0, 0]             grid_y shape [batch, 3, 76, 76]
            # tensor([[0., 0., 0., 0., 0.],
            #         [1., 1., 1., 1., 1.],
            #         [2., 2., 2., 2., 2.],
            #         [.., .., .., .., ..],
            #         [75., 75., 75., 75., 75.]])
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            # anchor_w  [batch_size, 3, fsize, fsize]
            # anchor_w[0][0], anchor box的宽除以strides后得到的值
            # tensor([[1.5000, 1.5000, ......, 1.5000, 1.5000],
            #         [1.5000, 1.5000, ......, 1.5000, 1.5000],
            #         [......, ......, ......, ......, ......],
            #         [1.5000, 1.5000, ......, 1.5000, 1.5000],
            #         [1.5000, 1.5000, ......, 1.5000, 1.5000]])
            # anchor_w[0][1]
            # tensor([[2.3750, 2.3750, ......, 2.3750, 2.3750],
            #         [2.3750, 2.3750, ......, 2.3750, 2.3750],
            #         [......, ......, ......, ......, ......],
            #         [2.3750, 2.3750, ......, 2.3750, 2.3750],
            #         [2.3750, 2.3750, ......, 2.3750, 2.3750]])
            # anchor_w[0][2]
            # tensor([[5., 5., .., 5., 5.],
            #         [5., 5., .., 5., 5.],
            #         [.., .., .., .., ..],
            #         [5., 5., .., 5., 5.],
            #         [5., 5., .., 5., 5.]])
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            # 同anchor_w，只不过这里是anchor box的高除以strides后得到的矩阵

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)

    def build_target(self, pred, labels, batchsize, fsize, n_ch, output_id):
        # pred: [batchsize, 3, fsize, fsize, 4]
        # labels(x1y1x2y2c): [batchsize, 60, 5]
        # target assignment
        tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device)
        obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device)
        tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device)
        target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device)

        # labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects, (batchsize, )  每张图片中有效检测框的数量

        truth_x_all = (labels[:, :, 2] + labels[:, :, 0]) / (self.strides[output_id] * 2) # gt中点的x坐标在输出特征图中的位置, [bs, 60]
        truth_y_all = (labels[:, :, 3] + labels[:, :, 1]) / (self.strides[output_id] * 2) # gt中点的y坐标在输出特征图中的位置, [bs, 60]
        truth_w_all = (labels[:, :, 2] - labels[:, :, 0]) / self.strides[output_id]  # gt的宽在输出特征图中的大小, [bs, 60]
        truth_h_all = (labels[:, :, 3] - labels[:, :, 1]) / self.strides[output_id]  # gt的高在输出特征图中的大小, [bs, 60]
        truth_i_all = truth_x_all.to(torch.int16).cpu().numpy()  # 向下取整
        truth_j_all = truth_y_all.to(torch.int16).cpu().numpy()

        for b in range(batchsize):
            n = int(nlabel[b])  # 该图片所拥有的gt数量
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4).to(self.device)
            truth_box[:n, 2] = truth_w_all[b, :n]  # [n, ]
            truth_box[:n, 3] = truth_h_all[b, :n]  # [n, ]
            truth_i = truth_i_all[b, :n]  # [n, ]
            truth_j = truth_j_all[b, :n]  # [n, ]

            # calculate iou between truth and reference anchors
            # truth_box：gt框在当前缩放比下的坐标及宽高
            # self.ref_anchors[output_id] 先验框在当前缩放比下的坐标及尺寸
            # array([[0., 0., 1.5, 2.],
            #        [0., 0., 2.375, 4.5],
            #        [0., 0., 5., 3.5],
            #        [0., 0., 4.5, 9.375],
            #        [0., 0., 9.5, 6.875],
            #        [0., 0., 9., 18.25],
            #        [0., 0., 17.75, 13.75],
            #        [0., 0., 24., 30.375],
            #        [0., 0., 57.375, 50.125]], dtype=float32)
            # 注意：这里检测框的左上角都是00，并且下面bboxes_iou的计算采用的是xyxy方式
            anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors[output_id], CIoU=True)  # (n, 9),

            # temp = bbox_iou(truth_box.cpu(), self.ref_anchors[output_id])

            best_n_all = anchor_ious_all.argmax(dim=1)  # (n, ) n个标签框中的每个标签框与哪个参考框最匹配
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_masks[output_id][0]) |  # 为True则表明这个标签框负责由该尺寸特征图的第beat_n个检测框预测
                           (best_n_all == self.anch_masks[output_id][1]) |  # 为False则表明这个标签框不由该尺寸特征图进行预测
                           (best_n_all == self.anch_masks[output_id][2]))

            if sum(best_n_mask) == 0:
                continue

            # truth_box中存放这该图片中n个检测框在当前特征图尺寸下的xywh
            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            # 将该特征图中预测出来的所有预测框与所有gt框计算IOU
            pred_ious = bboxes_iou(pred[b].view(-1, 4), truth_box, xyxy=False)  # pred: [batchsize, 3, fsize, fsize, 4]， truth_box：[n, 4]
            # pred_ious  [3*fsize*fsize, n]
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_thre)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])  # [3, fsize, fsize]  指定位置处的那个预测框与gt的最大IOU是否大于阈值0.5
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~ pred_best_iou  # 若True 指定位置处的预测框与gt的最大IOU小于阈值0.5  若F，则最大IOU大于阈值0.5

            for ti in range(best_n.shape[0]):  # 遍历当前图片中的每个gt框
                if best_n_mask[ti] == 1:  # 只有标签框与当前特征图对应的3个参考框匹配时
                    i, j = truth_i[ti], truth_j[ti]  # 当前标签框的中心落在哪个格子
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(
                        truth_w_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(
                        truth_h_all[b, ti] / torch.Tensor(self.masked_anchors[output_id])[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 4].to(torch.int16).cpu().numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)
        return obj_mask, tgt_mask, tgt_scale, target

    def forward(self, xin, labels=None):
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for output_id, output in enumerate(xin):  # output: [batchsize, 255, fsize, fsize]
            batchsize = output.shape[0]
            fsize = output.shape[2]
            n_ch = 5 + self.n_classes

            output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
            output = output.permute(0, 1, 3, 4, 2)  # .contiguous()  # [batchsize, 3, fsize, fsize, 85]

            # logistic activation for xy, obj, cls, [batchsize, 3, fsize, fsize, 83]，不对wh做sigmoid
            output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

            pred = output[..., :4].clone()          # pred: [batchsize, 3, fsize, fsize, 4]
            pred[..., 0] += self.grid_x[output_id]  # [batchsize, 3, fsize, fsize], sigmoid后的x加上全局x坐标, 预测框中心在fsize尺寸特征图中的位置，0-fsize
            pred[..., 1] += self.grid_y[output_id]  # [batchsize, 3, fsize, fsize], sigmoid后的y加上全局y坐标, 预测框中心在fsize尺寸特征图中的位置，0-fsize
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id]  # [batchsize, 3, fsize, fsize]，在fsize尺寸特征图下预测框的高
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id]  # [batchsize, 3, fsize, fsize]

            obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)

            # loss calculation
            # output      [batchsize, 3, fsize, fsize, 85]
            # target      [batchsize, 3, fsize, fsize, 85]
            # obj_mask    [batchsize, 3, fsize, fsize]
            # tgt_mask    [batchsize, 3, fsize, fsize, 84]
            # tgt_scale   [batchsize, 3, fsize, fsize, 2]
            output[..., 4] *= obj_mask  # (b, 3, f, f)
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask  # (b, 3, f, f, 84)
            output[..., 2:4] *= tgt_scale  # (b, 3, f, f, 2)

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            # target，output : [b, 3, f, f, 85]
            loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                              weight=tgt_scale * tgt_scale, reduction='sum')
            loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction='sum') / 2
            loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum')
            loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum')
            loss_l2 += F.mse_loss(input=output, target=target, reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls

        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2


def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def train(model, device, config, epochs=5, batch_size=1, save_cp=True, log_step=20, img_scale=0.5):
    train_dataset = Yolo_dataset(config.train_label, config, train=True)
    val_dataset = Yolo_dataset(config.val_label, config, train=False)

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate)

    val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=8,
                            pin_memory=True, drop_last=True, collate_fn=val_collate)

    writer = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}')
    # writer.add_images('legend',
    #                   torch.from_numpy(train_dataset.label2colorlegend2(cfg.DATA_CLASSES).transpose([2, 0, 1])).to(
    #                       device).unsqueeze(0))
    max_itr = config.TRAIN_EPOCHS * n_train
    # global_step = cfg.TRAIN_MINEPOCH * n_train
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained:
    ''')

    # learning rate setup
    def burnin_schedule(i):
        '相对原始学习率的系数'
        if i < config.burn_in: # 1000
            factor = pow(i / config.burn_in, 4)
        elif i < config.steps[0]: # 40W
            factor = 1.0
        elif i < config.steps[1]: # 45W
            factor = 0.1
        else:
            factor = 0.01
        return factor

    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate / config.batch,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate / config.batch,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    criterion = Yolo_loss(device=device, batch=config.batch // config.subdivisions, n_classes=config.classes)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, 0.001, 1e-6, 20)

    save_prefix = 'Yolov4_epoch'
    saved_models = deque()
    model.train()
    for epoch in range(epochs):
        # model.train()
        epoch_loss = 0
        epoch_step = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img', ncols=50) as pbar:
            for i, batch in enumerate(train_loader):
                global_step += 1
                epoch_step += 1
                images = batch[0]
                bboxes = batch[1]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)

                bboxes_pred = model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                # loss = loss / config.subdivisions
                loss.backward()

                epoch_loss += loss.item()

                if global_step % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if global_step % (log_step * config.subdivisions) == 0:
                    writer.add_scalar('train/Loss', loss.item(), global_step)
                    writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                    writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                    writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                    writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                    writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                    writer.add_scalar('lr', scheduler.get_lr()[0] * config.batch, global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                                        'loss_wh': loss_wh.item(),
                                        'loss_obj': loss_obj.item(),
                                        'loss_cls': loss_cls.item(),
                                        'loss_l2': loss_l2.item(),
                                        'lr': scheduler.get_lr()[0] * config.batch
                                        })
                    logging.debug('Train step_{}: loss : {},loss xy : {},loss wh : {},'
                                  'loss obj : {}，loss cls : {},loss l2 : {},lr : {}'
                                  .format(global_step, loss.item(), loss_xy.item(),
                                          loss_wh.item(), loss_obj.item(),
                                          loss_cls.item(), loss_l2.item(),
                                          scheduler.get_lr()[0] * config.batch))

                pbar.update(images.shape[0])

            if cfg.use_darknet_cfg:
                eval_model = Darknet(cfg.cfgfile, inference=True)
            else:
                eval_model = Yolov4(cfg.pretrained, n_classes=cfg.classes, inference=True)
            # eval_model = Yolov4(yolov4conv137weight=None, n_classes=config.classes, inference=True)
            if torch.cuda.device_count() > 1:
                eval_model.load_state_dict(model.module.state_dict())
            else:
                eval_model.load_state_dict(model.state_dict())
            eval_model.to(device)
            evaluator = evaluate(eval_model, val_loader, config, device)
            del eval_model

            stats = evaluator.coco_eval['bbox'].stats
            writer.add_scalar('train/AP', stats[0], global_step)
            writer.add_scalar('train/AP50', stats[1], global_step)
            writer.add_scalar('train/AP75', stats[2], global_step)
            writer.add_scalar('train/AP_small', stats[3], global_step)
            writer.add_scalar('train/AP_medium', stats[4], global_step)
            writer.add_scalar('train/AP_large', stats[5], global_step)
            writer.add_scalar('train/AR1', stats[6], global_step)
            writer.add_scalar('train/AR10', stats[7], global_step)
            writer.add_scalar('train/AR100', stats[8], global_step)
            writer.add_scalar('train/AR_small', stats[9], global_step)
            writer.add_scalar('train/AR_medium', stats[10], global_step)
            writer.add_scalar('train/AR_large', stats[11], global_step)

            if save_cp:
                try:
                    # os.mkdir(config.checkpoints)
                    os.makedirs(config.checkpoints, exist_ok=True)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                save_path = os.path.join(config.checkpoints, f'{save_prefix}{epoch + 1}.pth')
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.moduel,state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                logging.info(f'Checkpoint {epoch + 1} saved !')
                saved_models.append(save_path)
                if len(saved_models) > config.keep_checkpoint_max > 0:
                    model_to_remove = saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except:
                        logging.info(f'failed to remove {model_to_remove}')

    writer.close()


@torch.no_grad()
def evaluate(model, data_loader, cfg, device, logger=None, **kwargs):
    """ finished, tested
    """
    # cpu_device = torch.device("cpu")
    model.eval()
    # header = 'Test:'

    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')

    for images, targets in data_loader:
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(model_input)

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # outputs = outputs.cpu().detach().numpy()
        res = {}
        # for img, target, output in zip(images, targets, outputs):
        for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
            img_height, img_width = img.shape[:2]
            # boxes = output[...,:4].copy()  # output boxes in yolo format
            boxes = boxes.squeeze(2).cpu().detach().numpy()
            boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            boxes[...,0] = boxes[...,0]*img_width
            boxes[...,1] = boxes[...,1]*img_height
            boxes[...,2] = boxes[...,2]*img_width
            boxes[...,3] = boxes[...,3]*img_height
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # confs = output[...,4:].copy()
            confs = confs.cpu().detach().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = np.max(confs, axis=1).flatten()
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
    #                     help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='learning_rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-pretrained', type=str, default=None, help='pretrained yolov4.conv.137')
    parser.add_argument('-classes', type=int, default=80, help='dataset classes')
    parser.add_argument('-train_label_path', dest='train_label', type=str, default='train.txt', help="train label path")
    parser.add_argument(
        '-optimizer', type=str, default='adam',
        help='training optimizer',
        dest='TRAIN_OPTIMIZER')
    parser.add_argument(
        '-iou-type', type=str, default='iou',
        help='iou type (iou, giou, diou, ciou)',
        dest='iou_type')
    parser.add_argument(
        '-keep-checkpoint-max', type=int, default=10,
        help='maximum number of checkpoints to keep. If set 0, all checkpoints will be kept',
        dest='keep_checkpoint_max')
    args = vars(parser.parse_args())

    # for k in args.keys():
    #     cfg[k] = args.get(k)
    cfg.update(args)
    # 如果程序运行时命令行指定的参数中有参数和配置文件cfg中的重合，则用命令行中的新参数替换cfg中的旧参数
    # 如果不重合，则按配置文件中的参数来

    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def _get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')


if __name__ == "__main__":
    logging = init_logger(log_dir='log')
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if cfg.use_darknet_cfg:
        model = Darknet(cfg.cfgfile)
    else:
        model = Yolov4(cfg.pretrained, n_classes=cfg.classes)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:
        train(model=model,
              config=cfg,
              epochs=cfg.TRAIN_EPOCHS,
              device=device, )
    except KeyboardInterrupt:
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), 'INTERRUPTED.pth')
        else:
            torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
