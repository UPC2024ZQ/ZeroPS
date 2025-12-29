import json
import numpy as np
import os
from pytorch3d.io import IO
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_path", default="/mnt/sda/yuhengxue/reproduction_zerops_11_21", type=str
)
parser.add_argument(
    "--benchmark", default="AKBSeg", type=str
)
args = parser.parse_args()
benchmark = args.benchmark
base_path = args.base_path

if benchmark == "PartNetE":
    data_root = f"{base_path}/PartNetE"
    meta = json.load(open("PartNetE_meta.json"))
elif benchmark == 'AKBSeg':
    data_root = f"{base_path}/AKBSeg"
    meta = json.load(open("AKBSeg_meta.json", "r"))
else:
    raise ValueError('The benchmark does not exist!')

def calc_iou(pred, gt) -> float:
    I = np.logical_and(pred, gt).sum()
    U = np.logical_or(pred, gt).sum()
    iou = I / U * 100
    return iou

tot_iou = 0
category_iou_print = []


for category in meta.keys():
    print(f"==============={category}====================")
    models = os.listdir(f"{data_root}/{category}")  # list of models
    part_names = meta[category]
    pre_all_iou = []
    gt_cnt = []
    category_part_iou = 0
    for model in models:
        print(f"{model},", end="")
        # load gt label
        gt_part_label = np.load(
            f"{data_root}/{category}/{model}/label.npy", allow_pickle=True
        ).item()["instance_seg"]
        pre_part_label = np.load(
                f"{data_root}/{category}/{model}/unlabel_seg.npz", allow_pickle=True
            )["unlabel_seg"]
        gt_cnt.append(np.max(gt_part_label) + 1)
        for i in range(0, np.max(gt_part_label) + 1):
            part_max_iou = 0.0
            for k in np.unique(pre_part_label):
                if k == -1:
                    continue
                part_iou = calc_iou(pre_part_label == k, gt_part_label == i)
                if part_iou > part_max_iou:
                    part_max_iou = part_iou
            pre_all_iou.append(part_max_iou)
    print()
    category_part_iou = sum(pre_all_iou) / sum(gt_cnt)
    assert len(pre_all_iou) == sum(gt_cnt)
    print(category_part_iou)

    category_iou_print.append(category_part_iou)
    tot_iou += category_part_iou
print("=============================")
print(f"Average IoU: {tot_iou / len(meta.keys())}")


