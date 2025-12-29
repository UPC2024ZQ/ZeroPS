import json
import os
import shutil
import argparse
import h5py
import numpy as np

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

temp_data_name = f"temp/eval_{benchmark}_ins"


def load_gt_h5(fn):
    """Output: pts             B x N x 3   float32
            gt_mask         B x K x N   bool
            gt_mask_label   B x K       uint8
            gt_mask_valid   B x K       bool
            gt_mask_other   B x N       bool
    All the ground-truth masks are represented as 0/1 mask over the 10k point cloud.
    All the ground-truth masks are disjoint, complete and corresponding to unique semantic labels.
    Different test shapes have different numbers of ground-truth masks, they are at the top in array gt_mask indicated by gt_valid.
    """
    with h5py.File(fn, "r") as fin:
        gt_mask = fin["gt_mask"][:]
        gt_mask_label = fin["gt_mask_label"][:]
        gt_mask_valid = fin["gt_mask_valid"][:]
        gt_mask_other = fin["gt_mask_other"][:]
        return gt_mask, gt_mask_label, gt_mask_valid, gt_mask_other


def load_pred_h5(fn):
    """Output: mask    B x K x N   bool
            label   B x K       uint8
            valid   B x K       bool
            conf    B x K       float32
    We only evaluate on the part predictions with valid = True.
    We assume no pre-sorting according to confidence score.
    """
    with h5py.File(fn, "r") as fin:
        mask = fin["mask"][:]
        label = fin["label"][:]
        valid = fin["valid"][:]
        conf = fin["conf"][:]
        return mask, label, valid, conf


def check_mkdir(x):
    if os.path.exists(x):
        print("ERROR: folder %s exists! Please check and delete it!" % x)
        exit(1)
    else:
        os.mkdir(x)


def eval_per_class_ap(stat_fn, gt_dir, pred_dir, iou_threshold=0.5, plot_dir=None):
    """Input:  stat_fn contains all part ids and names
            gt_dir contains test-xx.h5
            pred_dir contains test-xx.h5
    Output: aps: Average Prediction Scores for each part category, evaluated on all test shapes
            mAP: mean AP
    """
    print("Evaluation Start.")
    print("Ground-truth Directory: %s" % gt_dir)
    print("Prediction Directory: %s" % pred_dir)

    if plot_dir is not None:
        check_mkdir(plot_dir)

    # read stat_fn
    # with open(stat_fn, 'r') as fin:
    #     part_name_list = [item.rstrip().split()[1] for item in fin.readlines()]
    # print('Part Name List: ', part_name_list)
    # n_labels = len(part_name_list)
    # print('Total Number of Semantic Labels: %d' % n_labels)

    part_name_list = stat_fn
    n_labels = len(part_name_list)

    # check all h5 files
    test_h5_list = []
    for item in os.listdir(gt_dir):
        if item.startswith("test-") and item.endswith(".h5"):
            if not os.path.exists(os.path.join(pred_dir, item)):
                print("ERROR: h5 file %s is in gt directory but not in pred directory.")
                exit(1)
            test_h5_list.append(item)

    # read each h5 file and collect per-part-category true_pos, false_pos and confidence scores
    true_pos_list = [[] for item in part_name_list]
    false_pos_list = [[] for item in part_name_list]
    conf_score_list = [[] for item in part_name_list]

    gt_npos = np.zeros((n_labels), dtype=np.int32)

    for item in test_h5_list:
        # print('Testing %s' % item)

        gt_mask, gt_mask_label, gt_mask_valid, gt_mask_other = load_gt_h5(
            os.path.join(gt_dir, item)
        )
        pred_mask, pred_label, pred_valid, pred_conf = load_pred_h5(
            os.path.join(pred_dir, item)
        )

        n_shape = gt_mask.shape[0]
        gt_n_ins = gt_mask.shape[1]
        pred_n_ins = pred_mask.shape[1]

        for i in range(n_shape):
            cur_pred_mask = pred_mask[i, ...]
            cur_pred_label = pred_label[i, :]
            cur_pred_conf = pred_conf[i, :]
            cur_pred_valid = pred_valid[i, :]

            cur_gt_mask = gt_mask[i, ...]
            cur_gt_label = gt_mask_label[i, :]
            cur_gt_valid = gt_mask_valid[i, :]
            cur_gt_other = gt_mask_other[i, :]

            # classify all valid gt masks by part categories
            gt_mask_per_cat = [[] for item in part_name_list]
            for j in range(gt_n_ins):
                if cur_gt_valid[j]:
                    sem_id = cur_gt_label[j]
                    gt_mask_per_cat[sem_id].append(j)
                    gt_npos[sem_id] += 1
                    # print(f'{item}')

            # sort prediction and match iou to gt masks
            cur_pred_conf[~cur_pred_valid] = 0.0
            order = np.argsort(-cur_pred_conf)

            gt_used = np.zeros((gt_n_ins), dtype=bool)

            for j in range(pred_n_ins):
                idx = order[j]
                if cur_pred_valid[idx]:
                    sem_id = cur_pred_label[idx]

                    iou_max = 0.0
                    cor_gt_id = -1
                    for k in gt_mask_per_cat[sem_id]:
                        if not gt_used[k]:
                            # Remove points with gt label *other* from the prediction
                            # We will not evaluate them in the IoU since they can be assigned any label
                            clean_cur_pred_mask = cur_pred_mask[idx, :] & (
                                ~cur_gt_other
                            )

                            intersect = np.sum(cur_gt_mask[k, :] & clean_cur_pred_mask)
                            union = np.sum(cur_gt_mask[k, :] | clean_cur_pred_mask)
                            iou = intersect * 1.0 / union

                            if iou > iou_max:
                                iou_max = iou
                                cor_gt_id = k

                    if iou_max > iou_threshold:
                        gt_used[cor_gt_id] = True

                        # add in a true positive
                        true_pos_list[sem_id].append(True)
                        false_pos_list[sem_id].append(False)
                        conf_score_list[sem_id].append(cur_pred_conf[idx])
                    else:
                        # add in a false positive
                        true_pos_list[sem_id].append(False)
                        false_pos_list[sem_id].append(True)
                        conf_score_list[sem_id].append(cur_pred_conf[idx])

    # compute per-part-category AP
    aps = np.zeros((n_labels), dtype=np.float32)
    ap_valids = np.ones((n_labels), dtype=bool)
    for i in range(n_labels):
        has_pred = len(true_pos_list[i]) > 0
        has_gt = gt_npos[i] > 0

        if not has_gt:
            ap_valids[i] = False
            continue

        if has_gt and not has_pred:
            continue

        cur_true_pos = np.array(true_pos_list[i], dtype=np.float32)
        cur_false_pos = np.array(false_pos_list[i], dtype=np.float32)
        cur_conf_score = np.array(conf_score_list[i], dtype=np.float32)

        # sort according to confidence score again
        order = np.argsort(-cur_conf_score)
        sorted_true_pos = cur_true_pos[order]
        sorted_false_pos = cur_false_pos[order]

        out_plot_fn = None
        if plot_dir is not None:
            out_plot_fn = os.path.join(
                plot_dir, part_name_list[i].replace("/", "-") + ".png"
            )

        aps[i] = compute_ap(
            sorted_true_pos, sorted_false_pos, gt_npos[i], plot_fn=out_plot_fn
        )

    # compute mean AP
    mean_ap = np.sum(aps * ap_valids) / np.sum(ap_valids)

    return aps * 100, ap_valids, gt_npos, mean_ap * 100


def compute_ap(tp, fp, gt_npos, n_bins=100, plot_fn=None):
    assert len(tp) == len(
        fp
    ), "ERROR: the length of true_pos and false_pos is not the same!"

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    rec = tp / gt_npos
    prec = tp / (fp + tp)

    rec = np.insert(rec, 0, 0.0)
    prec = np.insert(prec, 0, 1.0)

    ap = 0.0
    delta = 1.0 / n_bins

    out_rec = np.arange(0, 1 + delta, delta)
    out_prec = np.zeros((n_bins + 1), dtype=np.float32)

    for idx, t in enumerate(out_rec):
        prec1 = prec[rec >= t]
        if len(prec1) == 0:
            p = 0.0
        else:
            p = max(prec1)

        out_prec[idx] = p
        ap = ap + p / (n_bins + 1)

    if plot_fn is not None:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(out_rec, out_prec, "b-")
        plt.title("PR-Curve (AP: %4.2f%%)" % (ap * 100))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        fig.savefig(plot_fn)
        plt.close(fig)

    return ap


def gen_single_shape_gt_h5(category, model, meta, data_root):
    # sem_label, ins_label = get_gt_labels(category, model, meta[category], data_root)
    # hf = h5py.File('/media/minghua/data/eval_results/rendering/%s/%s/gt_ins.h5' % (category, model), 'w')

    label = np.load(
        f"{data_root}/{category}/{model}/label.npy", allow_pickle=True
    ).item()
    sem_label = label["semantic_seg"]
    ins_label = label["instance_seg"]
    hf = h5py.File(f"{temp_data_name}/gt/test-%s.h5" % (model), "w")
    gt_mask = []
    gt_mask_label = []
    gt_mask_valid = []
    gt_mask_other = []
    for ins in np.unique(ins_label):
        if ins == -1:
            continue
        gt_mask.append(ins_label == ins)
        idx = np.where(ins_label == ins)[0]
        gt_mask_label.append(sem_label[idx[0]])
        gt_mask_valid.append(True)
    # temp = np.expand_dims(np.hstack(gt_mask))
    if len(gt_mask) != 0:
        gt_mask = np.expand_dims(gt_mask, axis=0)
        gt_mask_label = np.array(gt_mask_label, dtype=np.uint8).reshape(1, -1)
        gt_mask_valid = np.array(gt_mask_valid).reshape(1, -1)
        gt_mask_other = np.zeros((1, sem_label.shape[0]), dtype=bool)
    else:
        # skip the several blank gt in partnete
        os.remove(f"{temp_data_name}/gt/test-{model}.h5")
        return False
        

    # gt_mask = np.expand_dims(gt_mask, axis=0)
    # gt_mask_label = np.array(gt_mask_label, dtype=np.uint8).reshape(1, -1)
    # gt_mask_valid = np.array(gt_mask_valid).reshape(1, -1)
    # gt_mask_other = np.zeros((1, sem_label.shape[0]), dtype=bool)

    hf.create_dataset("gt_mask", data=gt_mask)
    hf.create_dataset("gt_mask_label", data=gt_mask_label)
    hf.create_dataset("gt_mask_valid", data=gt_mask_valid)
    hf.create_dataset("gt_mask_other", data=gt_mask_other)
    hf.close()
    return True


def gen_single_shape_pred_h5_new(category, model, meta, data_root):
    hf = h5py.File(f"{temp_data_name}/pred/test-%s.h5" % (model), "w")
    mask_list = []
    label_list = []
    valid_list = []
    conf_list = []
    pred_sem_label = np.load(
        f"{data_root}/{category}/{model}/ins_sem_seg.npz",
        allow_pickle=True,
    )["sem_label"]
    pred_ins_label = np.load(
        f"{data_root}/{category}/{model}/ins_sem_seg.npz",
        allow_pickle=True,
    )["ins_label"]

    for ins in np.unique(pred_ins_label):
        if ins == -1:
            continue
        mask_list.append(pred_ins_label == ins)
        label_list.append(pred_sem_label[np.where(pred_ins_label == ins)[0][0]])
        valid_list.append(True)
        conf_list.append(1.0)
    if len(mask_list) != 0:
        mask_list = np.expand_dims(mask_list, axis=0)
        label_list = np.array(label_list, dtype=np.uint8).reshape(1, -1)
        valid_list = np.array(valid_list).reshape(1, -1)
        conf_list = np.array(conf_list, dtype=np.float32).reshape(1, -1)
    else:
        # import pdb;pdb.set_trace()
        mask_list = np.zeros((1, 0, 0), dtype=bool)
        label_list = np.zeros((1, 0), dtype=np.uint8)
        valid_list = np.zeros((1, 0), dtype=bool)
        conf_list = np.zeros((1, 0), dtype=np.float32)

    hf.create_dataset("mask", data=mask_list)
    hf.create_dataset("label", data=label_list)
    hf.create_dataset("valid", data=valid_list)
    hf.create_dataset("conf", data=conf_list)
    hf.close()


if __name__ == "__main__":

    gt_dir = f"{temp_data_name}/gt"
    pred_dir = f"{temp_data_name}/pred"
    if os.path.exists(gt_dir):
        shutil.rmtree(gt_dir)
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)

    aps_record = []
    categories = meta.keys()
    for category in categories:
        # save temp gt pred dir
        # check there is any file in the dir
        if os.path.exists(gt_dir):
            ValueError(f"{gt_dir} exists!")
        if os.path.exists(pred_dir):
            ValueError(f"{pred_dir} exists!")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        print(f"==============={category}====================")
        models = os.listdir("%s/%s/" % (data_root, category))
        for model in models:
            print(f"{model}, ", end="")
            check = gen_single_shape_gt_h5(category, model, meta, data_root)
            if check:
                gen_single_shape_pred_h5_new(category, model, meta, data_root)
        aps, ap_valids, gt_npos, mean_ap = eval_per_class_ap(
            meta[category], gt_dir, pred_dir, 0.5
        )
        aps_record.append(aps)
        print(aps)
        # clear gt pred dir
        shutil.rmtree(gt_dir)
        shutil.rmtree(pred_dir)
    shutil.rmtree(temp_data_name)

    all_mean_ap = 0
    for i, category in enumerate(categories):
        all_mean_ap += aps_record[i].mean()
    print(f"mAP50: {all_mean_ap / len(categories)}")
