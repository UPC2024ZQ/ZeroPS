import json
import numpy as np
import os
import open3d as o3d
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

categories = meta.keys()
part_miou_record = []
for category in categories:
    print(f"==============={category}====================")
    models = os.listdir(f"{data_root}/{category}")  # list of models
    part_names = meta[category]
    cnt = np.zeros(len(part_names))
    cnt_iou = np.zeros(len(part_names))

    for model in models:
        print(f"{model},", end="")
        # load gt label
        gt_sem_label = np.load(
            f"{data_root}/{category}/{model}/label.npy", allow_pickle=True
        ).item()["semantic_seg"]
        sem_pred = np.load(
            f"{data_root}/{category}/{model}/ins_sem_seg.npz", allow_pickle=True
        )["sem_label"]
        for i, part in enumerate(part_names):
            if (gt_sem_label == i).sum() == 0:
                continue
            iou = calc_iou(sem_pred == i, gt_sem_label == i)
            cnt[i] += 1
            cnt_iou[i] += iou

    part_miou = cnt_iou / cnt
    part_miou_record.append(part_miou)
    print()
    print(part_miou)

all_miou = 0
for i, category in enumerate(categories):
    all_miou += part_miou_record[i].mean()
print(f"miou: {all_miou / len(categories)}")
