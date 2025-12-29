import cv2
from tqdm import tqdm
from pointnet2_ops import pointnet2_utils
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.structures.utils import list_to_padded
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor,
)
from pytorch3d.io import IO
from pytorch3d.ops import sample_farthest_points, knn_points
import open3d as o3d
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import argparse
import json
import os
import time
import torch
import copy
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

# =========== hyper-parameter =================
parser = argparse.ArgumentParser(description="Hyper-parameters")
parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for computation")
parser.add_argument('--unlabeled_seg', action="store_true", help="Whether to unlabeled seg")
parser.add_argument('--T_merge', type=float, default=0.3, help="Threshold for merging")
parser.add_argument('--with_labeling', action='store_true', help="Whether to use labeling")
parser.add_argument('--min_points', type=int, default=3, help="【DBSCAN for 3D Largest Connected Component】Minimum points")
parser.add_argument('--eps', type=float, default=0.007, help="【DBSCAN for 3D Largest Connected Component】Epsilon")
args = parser.parse_args()
device = torch.device(args.device)
unlabeled_seg = args.unlabeled_seg
with_labeling = args.with_labeling
T_merge = args.T_merge
min_points = args.min_points
eps = args.eps
# =========== hyper-parameter =================

# =========== load sam model =================
if unlabeled_seg:
    print("loading sam-h model...")
    sam = sam_model_registry["vit_h"](checkpoint="./models/sam_vit_h_4b8939.pth").to(
        device=device
    )
    sam.to(device=device)
    print("loading sam-h model...done")
# =========== load sam model =================

# =========== load glip model =================

if with_labeling:
    print("loading GLIP model...done")
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file("GLIP/configs/pretrain/glip_Swin_L.yaml")
    cfg.merge_from_list(["MODEL.WEIGHT", "models/glip_large_model.pth"])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    glip_demo = GLIPDemo(
        cfg, min_image_size=800, confidence_threshold=0.7, show_mask_heatmaps=False
    )
    print("loading GLIP model...done")
# =========== load glip model =================

view_name = [
    "v0",
    "v0_1",
    "v1",
    "v1_2",
    "v2",
    "v2_3",
    "v3",
    "v0_3",
    "v0_5",
    "v1_6",
    "v2_7",
    "v3_4",
    "v5",
    "v5_6",
    "v6",
    "v6_7",
    "v7",
    "v4_7",
    "v4",
    "v4_5",
]

view_adjacency_table = {
    "v0": ["v0_1", "v0_3", "v0_5"],
    "v1": ["v0_1", "v1_2", "v1_6"],
    "v2": ["v1_2", "v2_3", "v2_7"],
    "v3": ["v0_3", "v2_3", "v3_4"],
    "v4": ["v3_4", "v4_5", "v4_7"],
    "v5": ["v0_5", "v4_5", "v5_6"],
    "v6": ["v1_6", "v5_6", "v6_7"],
    "v7": ["v2_7", "v4_7", "v6_7"],
    "v0_1": ["v0", "v1"],
    "v0_3": ["v0", "v3"],
    "v0_5": ["v0", "v5"],
    "v1_2": ["v1", "v2"],
    "v1_6": ["v1", "v6"],
    "v2_3": ["v2", "v3"],
    "v2_7": ["v2", "v7"],
    "v3_4": ["v3", "v4"],
    "v4_5": ["v4", "v5"],
    "v4_7": ["v4", "v7"],
    "v5_6": ["v5", "v6"],
    "v6_7": ["v6", "v7"],
}

def bfs_view_sequence(adjacency_list, start_node):
    queue = [start_node]
    visited = set()
    extend_sequence = []
    while queue:
        current_node = queue.pop(0)
        if current_node not in visited:
            extend_sequence.append(current_node)
            visited.add(current_node)

            neighbors = adjacency_list[current_node]
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
    return extend_sequence

def knn_propagation_background(unlabel_seg, xyz, bg_label=-1):
    assert unlabel_seg.shape[0] == xyz.shape[0]
    bg_idx = np.where(unlabel_seg == bg_label)[0]
    fg_idx = np.where(unlabel_seg != bg_label)[0]
    nn_idx = (
        knn_points(
            torch.tensor(xyz[bg_idx]).unsqueeze(0).to(device),
            torch.tensor(xyz[fg_idx]).unsqueeze(0).to(device),
            K=1,
        )[1]
        .cpu()
        .numpy()
        .squeeze()
    )
    unlabel_seg[bg_idx] = unlabel_seg[fg_idx[nn_idx]]
    return unlabel_seg


def visualize_groups_3d(xyz, components, save_dir, name):
    rgb = np.zeros_like(xyz)
    for i in range(len(components)):
        if np.size(components[i]) == 0:
            continue
        # Generate a random color with increased brightness and strong contrast
        # Random float between 0.3 and 1.0 for bright colors
        color = np.random.rand(3) * 0.7 + 0.3
        # Use complementary colors to enhance contrast
        rgb[components[i], :] = 1.0 - color
    save_colored_pc(os.path.join(save_dir, name), xyz, rgb)


def save_colored_pc(file_name, xyz, rgb):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(file_name, pc)


def normalize_pc_o3d(pc_file):
    pc = o3d.io.read_point_cloud(pc_file)
    xyz = np.asarray(pc.points)
    rgb = np.asarray(pc.colors)
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / np.linalg.norm(xyz, axis=1, ord=2).max()
    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.float32)
    return xyz, rgb


def get_largest_connected_domain(mask):
    mask_cv = mask.astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_cv, connectivity=4
    )
    if stats.shape[0] == 1:
        return mask
    largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    mask_after_filter = np.zeros((mask_cv.shape[0], mask_cv.shape[1]), np.uint8)
    largest_component_mask = labels == largest_component_index
    mask_after_filter[largest_component_mask] = 1
    return mask_after_filter != 0


def get_lcd(xyz, n=1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress=False))
    labels_unique, counts = np.unique(labels, return_counts=True)
    if np.size(labels_unique) == 1 and labels_unique[0] == -1:
        return None
    if labels_unique[0] == -1:
        labels_unique = labels_unique[1:]
        counts = counts[1:]
    all_idx = []
    while n > 0:
        labels_idx = np.argmax(counts)
        most_common_index = np.where(labels == labels_unique[labels_idx])[0]
        all_idx.extend(most_common_index.tolist())
        counts[labels_idx] = -1
        n -= 1
    return np.array(all_idx)


def compute_centroid(point_cloud):
    point_cloud = np.asarray(point_cloud)
    num_points = point_cloud.shape[0]
    sum_xyz = np.sum(point_cloud, axis=0)
    centroid_xyz = sum_xyz / num_points
    distances = np.linalg.norm(point_cloud - centroid_xyz, axis=1)
    min_index = np.argmin(distances)
    return min_index


def point_feature_mapping_view(pc_idx, feature):
    # pc_idx shape (resolution, resolution)
    # feature shape (num_points, num_features)
    image = feature[pc_idx]
    # background color
    image[pc_idx == -1] = np.array([1, 1, 1])
    image = (image * 255).astype(np.uint8)
    return image


def generate_groups_3d_starting(
    starting_view_name,
    view_name,
    sam_predictor,
    xyz,
    rgb,
    pc_idx,
    screen_coords,
    device,
    save_dir,
):
    groups_3d_starting = []
    starting_view_idx = view_name.index(starting_view_name)
    image = point_feature_mapping_view(pc_idx[starting_view_idx], feature=rgb)
    groups_2d = generate_groups_2d(
        image,
        xyz,
        sam_predictor,
        pc_idx[starting_view_idx].squeeze(),
        screen_coords[starting_view_idx].squeeze(),
        device,
    )
    for group_id in range(0, np.max(groups_2d) + 1):
        group_3d = np.unique(pc_idx[starting_view_idx].squeeze()[group_id == groups_2d])
        group_3d = group_3d[group_3d != -1]
        if np.size(group_3d) == 0:
            continue
        group_3d_remo = get_lcd(xyz[group_3d])
        if group_3d_remo is None:
            continue
        group_3d_remo = group_3d[group_3d_remo]
        groups_3d_starting.append(group_3d_remo)
    return groups_3d_starting


def single_view_extension(
    groups_3d, pc_idx_v, screen_coord_v, xyz, sam_encoder, device
):
    all_g3d_vis_pts_idx = []  # prompt points for each group in the vi visual field
    for g3d in groups_3d:
        g3d_vis_pts_idx = g3d[np.isin(g3d, pc_idx_v)]
        if np.size(g3d_vis_pts_idx) == 0:
            all_g3d_vis_pts_idx.append(None)
            continue
        g3d_vi_prompt_proportion = np.count_nonzero(g3d_vis_pts_idx) / np.count_nonzero(
            g3d
        )
        if g3d_vi_prompt_proportion < 0.05:
            all_g3d_vis_pts_idx.append(None)
        else:
            all_g3d_vis_pts_idx.append(g3d_vis_pts_idx)
    if all(x is None for x in all_g3d_vis_pts_idx):
        return groups_3d, None

    # FPS 3
    fps_input_pts = [
        torch.from_numpy(xyz[g3d_vis_pts_idx])
        for g3d_vis_pts_idx in all_g3d_vis_pts_idx
        if g3d_vis_pts_idx is not None
    ]
    fps_input_pts_lens = torch.tensor([len(p) for p in fps_input_pts])
    _, prompt_pts_3d_fps_idxs = sample_farthest_points(
        points=list_to_padded(
            fps_input_pts,
            (int(fps_input_pts_lens.max()), 3),
            pad_value=0.0,
            equisized=False,
        ),
        lengths=fps_input_pts_lens,
        K=3,
    )
    prompt_pts_3d_fps_idxs = list(prompt_pts_3d_fps_idxs.numpy())
    prompt_pts_3d_fps_idxs = np.asarray(
        [
            g3d_vis_pts_idx[prompt_pts_3d_fps_idxs.pop(0)]
            for g3d_vis_pts_idx in all_g3d_vis_pts_idx
            if g3d_vis_pts_idx is not None
        ]
    )

    prompt_pts_3d_cent_idx = np.array(
        [
            g3d_vis_pts_idx[compute_centroid(xyz[g3d_vis_pts_idx])]
            for g3d_vis_pts_idx in all_g3d_vis_pts_idx
            if g3d_vis_pts_idx is not None
        ]
    )[:, None]

    prompt_pts_3d_all_idx = np.hstack((prompt_pts_3d_fps_idxs, prompt_pts_3d_cent_idx))
    prompt_pts_2d_all_idx = screen_coord_v[prompt_pts_3d_all_idx]

    # predict and extend
    point_coords = sam_encoder.transform.apply_coords(
        prompt_pts_2d_all_idx,
        sam_encoder.original_size,
    )
    in_points = torch.as_tensor(point_coords, device=device)
    in_labels = torch.ones(
        (in_points.shape[0], in_points.shape[1]), dtype=torch.int, device=device
    )
    masks, iou_preds, _ = sam_encoder.predict_torch(
        in_points, in_labels, multimask_output=True, return_logits=False
    )
    masks = masks.cpu().numpy()
    iou_preds = iou_preds.cpu().numpy()
    all_score = []
    idx = 0
    for gId, g3d_vis_pts_idx in enumerate(all_g3d_vis_pts_idx):
        if g3d_vis_pts_idx is None:
            continue
        group_mask = masks[idx].squeeze()
        score = iou_preds[idx].squeeze()
        idx += 1
        min_area_idx = np.argmin(np.sum(group_mask, axis=(1, 2)))
        min_area_mask = group_mask[min_area_idx]
        if score[min_area_idx] < 0.8:
            continue
        all_score.append(float(score[min_area_idx]))
        # extend 3D area
        group_3d_ext = np.unique(pc_idx_v[min_area_mask])
        group_3d_ext = group_3d_ext[group_3d_ext != -1]
        if np.size(group_3d_ext) == 0:
            continue
        group_3d_ext_remo = get_lcd(xyz[group_3d_ext])
        if group_3d_ext_remo is None:
            continue
        group_3d_ext = group_3d_ext[group_3d_ext_remo]
        intersect_pts = np.intersect1d(groups_3d[gId], group_3d_ext)
        is_union = np.size(intersect_pts) != 0
        if not is_union:
            continue
        # union
        groups_3d[gId] = np.sort(np.union1d(groups_3d[gId], group_3d_ext))
    # mask_mean_score = np.mean(np.array(all_score))
    return groups_3d, None


def generate_groups_2d(image, xyz, sam_predictor, image_3d_idx, screen_coord, device):
    # obj_visible_points  fps
    vis_3d_idx = np.unique(image_3d_idx)[1:]

    # pointnet2_utils
    # vis_3d_idx_sub = pointnet2_utils.furthest_point_sample(torch.from_numpy(xyz[vis_3d_idx]).unsqueeze(
    #     0).to(device), 256).cpu().squeeze(0).numpy()
    # vis_3d_idx_sub = vis_3d_idx[vis_3d_idx_sub]

    # pytorch3d
    _, vis_3d_idx_sub = sample_farthest_points(
        torch.from_numpy(xyz[vis_3d_idx])[None, ...].to(device), K=256
    )
    vis_3d_idx_sub = vis_3d_idx[vis_3d_idx_sub[0].cpu().numpy()]

    # Get the prompt coordinates and batch number
    point_grids = screen_coord[vis_3d_idx_sub]
    point_grids = point_grids / image.shape[0]  # normalized to [0,1]
    point_grids = [np.array(point_grids)]

    points_per_batch = min(point_grids[0].shape[0], 80)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=None,
        points_per_batch=points_per_batch,
        min_mask_region_area=image.shape[0] / 4,
        point_grids=point_grids,
    )
    mask_generator.predictor = sam_predictor
    masks = mask_generator.generate(image)

    obj_area = np.count_nonzero(image_3d_idx != -1)
    masks = list(
        filter(
            lambda mask: (mask["area"] > image.shape[0] / 4)
            and (mask["area"] <= obj_area * 0.98),
            masks,
        )
    )

    # grouping
    masks = sorted(masks, key=lambda mask: mask["area"], reverse=False)
    group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    num_masks = len(masks)
    group_counter = 0
    for i in reversed(range(num_masks)):
        group_ids[masks[i]["segmentation"]] = group_counter
        group_counter += 1

    # only the largest connected domain is kept
    group_ids_temp = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    for group_id in range(np.max(group_ids), -1, -1):
        group_2d_mask = group_ids == group_id
        group_2d_mask_filtering = get_largest_connected_domain(group_2d_mask)
        group_ids_temp[group_2d_mask_filtering] = group_id
    group_ids[:] = group_ids_temp

    return group_ids


def check_pc_within_bbox(x1, y1, x2, y2, pc):
    flag = np.logical_and(pc[:, 0] > x1, pc[:, 0] < x2)
    flag = np.logical_and(flag, pc[:, 1] > y1)
    flag = np.logical_and(flag, pc[:, 1] < y2)
    return flag


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"] + 1e-6
    assert bb1["y1"] < bb1["y2"] + 1e-6
    assert bb2["x1"] < bb2["x2"] + 1e-6
    assert bb2["y1"] < bb2["y2"] + 1e-6

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_iou_mask(matrix1, matrix2):
    intersection = np.logical_and(matrix1, matrix2)
    union = np.logical_or(matrix1, matrix2)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou


def labeling(
    model_dir,
    xyz,
    rgb,
    pc_idx,
    screen_coords,
    unlabel_seg,
    part_names,
    num_views=20,
    save_pred_img=False,
):
    parts = []
    for i in np.unique(unlabel_seg):
        if i == -1:
            continue
        parts.append(np.where(unlabel_seg == i)[0])

    if len(parts) == 0:
        return

    if save_pred_img:
        pred_dir = os.path.join(model_dir, "preds")
        os.makedirs(pred_dir, exist_ok=True)

    predictions = []
    for i in range(num_views):
        image = point_feature_mapping_view(pc_idx[i], feature=rgb)
        result, top_predictions = glip_demo.run_on_web_image(image, part_names, 0.5)
        if save_pred_img:
            plt.imsave(f'{pred_dir}/{i}.png', result[:, :, [2, 1, 0]])
        bbox = top_predictions.bbox.cpu().numpy()
        score = top_predictions.get_field("scores").cpu().numpy()
        labels = top_predictions.get_field("labels").cpu().numpy()
        for j in range(len(bbox)):
            x1, y1, x2, y2 = bbox[j].tolist()
            ratio = check_pc_within_bbox(x1, y1, x2, y2, screen_coords[i]).mean()
            if ratio > 0.98:
                continue
            predictions.append(
                {
                    "image_id": i,
                    "category_id": labels[j].item(),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score[j].item(),
                }
            )

    # two-dimensional checking mechanism , vote matrix
    vote_martix = np.zeros((len(parts), len(part_names)))
    for i, pre in enumerate(predictions):
        x, y, w, h = pre["bbox"]
        pre_bbox = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
        # pre 2d->3d
        pre_3d = np.unique(
            pc_idx[pre["image_id"]].squeeze()[int(y) : int(y + h), int(x) : int(x + w)]
        )[1:]
        parts_score_2d = []
        parts_score_3d = []
        for part in parts:
            # score 2d
            # part 3D->2d->bbox
            part_2d = screen_coords[pre["image_id"]][part]
            if np.size(part_2d) == 0:
                parts_score_2d.append(0.0)
            else:
                part_bbox = {
                    "x1": np.min(part_2d[:, 0]),
                    "y1": np.min(part_2d[:, 1]),
                    "x2": np.max(part_2d[:, 0]),
                    "y2": np.max(part_2d[:, 1]),
                }
                parts_score_2d.append(get_iou(pre_bbox, part_bbox))
            # score 3d
            intersect = np.intersect1d(part, pre_3d)
            if np.size(intersect) == 0:
                parts_score_3d.append(0.0)
            else:
                parts_score_3d.append(
                    np.size(intersect) / np.size(np.union1d(part, pre_3d))
                )
        part_idx_2d = parts_score_2d.index(max(parts_score_2d))
        part_idx_3d = parts_score_3d.index(max(parts_score_3d))
        if part_idx_2d != part_idx_3d:
            continue
        part_idx = part_idx_2d
        vote_martix[part_idx, pre["category_id"] - 1] += 1

    # Class Non-highest Vote Penalty , decision matrix
    max_labels = np.max(vote_martix, axis=0)
    max_labels[max_labels == 0] = 1
    w_votes = vote_martix / max_labels
    w_votes = np.where((w_votes >= 0) & (w_votes < 0.5), 0, w_votes)
    w_votes = np.where((w_votes >= 0.5) & (w_votes < 1), 0.5, w_votes)
    decision_matrix = w_votes * vote_martix
    max_indices = np.argmax(decision_matrix, axis=1)
    max_value = decision_matrix[np.arange(len(decision_matrix)), max_indices]
    ins_label = np.where(max_value != 0, max_indices, np.nan)

    sem_seg = np.ones(xyz.shape[0], dtype=int) * -1
    ins_seg = np.ones(xyz.shape[0], dtype=int) * -1
    ins_cnt = 0
    for i in range(len(part_names)):
        part_idx = np.where(ins_label == i)[0]
        sem_idx = []
        for j in part_idx:
            ins_seg[parts[j]] = ins_cnt
            ins_cnt += 1
            sem_idx.extend(parts[j])
        sem_seg[sem_idx] = i
    return sem_seg, ins_seg


def render_single_view(
    pc,
    view,
    device,
    # background_color=(1, 1, 1),
    resolution=800,
    camera_distance=2.2,
    point_size=0.005,
    points_per_pixel=1,
    bin_size=0,
    znear=0.01,
):
    R, T = look_at_view_transform(camera_distance, view[0], view[1], device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear)

    raster_settings = PointsRasterizationSettings(
        image_size=resolution,
        radius=point_size,
        points_per_pixel=points_per_pixel,
        bin_size=bin_size,
    )
    # compositor = NormWeightedCompositor(background_color=background_color).to(device)

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings).to(
        device
    )
    # renderer = PointsRenderer(
    #     rasterizer=rasterizer,
    #     compositor=compositor,
    # ).to(device)
    # import pdb; pdb.set_trace()
    # img = renderer(pc)
    pc_idx = rasterizer(pc).idx
    screen_coords = cameras.transform_points_screen(
        pc._points_list[0], image_size=(resolution, resolution)
    )
    return None, pc_idx, screen_coords


def render_pc(xyz, rgb, device, resolution=800):
    pc = Pointclouds(
        points=[torch.Tensor(xyz).to(device)], features=[torch.Tensor(rgb).to(device)]
    )

    v0_elev = v0_azim = 35
    views = [
        [v0_elev, -v0_azim],  # v0
        [v0_elev, -v0_azim + 45],  # v0_1
        [v0_elev, -v0_azim + 90],  # v1
        [v0_elev, -v0_azim + 135],  # v1_2
        [v0_elev, -v0_azim + 180],  # v2
        [v0_elev, -v0_azim + 225],  # v2_3
        [v0_elev, -v0_azim + 270],  # v3
        [v0_elev, -v0_azim + 315],  # v0_3
        [v0_elev - 45, -v0_azim],  # v0_5
        [v0_elev - 45, -v0_azim + 90],  # v1_6
        [v0_elev - 45, -v0_azim + 180],  # v2_7
        [v0_elev - 45, -v0_azim + 270],  # v3_4
        [v0_elev - 90, -v0_azim],  # v5
        [v0_elev - 90, -v0_azim + 45],  # v5_6
        [v0_elev - 90, -v0_azim + 90],  # v6
        [v0_elev - 90, -v0_azim + 135],  # v6_7
        [v0_elev - 90, -v0_azim + 180],  # v7
        [v0_elev - 90, -v0_azim + 225],  # v4_7
        [v0_elev - 90, -v0_azim + 270],  # v4
        [v0_elev - 90, -v0_azim + 315],  # v4_5
    ]

    pc_idx_list = []
    screen_coords_list = []
    progress_bar = tqdm(total=len(views), desc="rendering...")
    for i, view in enumerate(views):
        img, pc_idx, screen_coords = render_single_view(
            pc, view, device, resolution=resolution
        )
        pc_idx_list.append(pc_idx)
        screen_coords_list.append(screen_coords)
        progress_bar.update(1)
    progress_bar.close()
    pc_idx = torch.cat(pc_idx_list, dim=0).squeeze()
    screen_coords = torch.cat(screen_coords_list, dim=0).reshape(len(views), -1, 3)[
        ..., :2
    ]
    return pc_idx.cpu().numpy(), screen_coords.cpu().numpy()


def merging(groups_3d, T_merge, xyz):
    print(f"merging ...  |||| T_merge:{T_merge}")
    all_groups_3d = []
    for starting_view_name in groups_3d.keys():
        for gid in range(0, len(groups_3d[starting_view_name])):
            all_groups_3d.append(groups_3d[starting_view_name][gid])
    all_groups_3d.sort(key=len, reverse=True)
    all_groups_3d = list(filter(lambda group: np.size(group) > 0, all_groups_3d))
    all_groups_3d_merged = []
    for i, group_3d in enumerate(all_groups_3d):
        flag = False
        for j, group_3d_merged in enumerate(all_groups_3d_merged):
            intersect_pts = np.intersect1d(group_3d_merged, group_3d)
            union1d_pts = np.union1d(group_3d_merged, group_3d)
            iou = np.size(intersect_pts) / np.size(union1d_pts)
            if iou > T_merge:
                all_groups_3d_merged[j] = np.unique(
                    np.concatenate((group_3d_merged, group_3d))
                )
                flag = True
                break
        if not flag:
            all_groups_3d_merged.append(group_3d)

    all_groups_3d_ = np.ones(xyz.shape[0], dtype=int) * -1
    for i, m in enumerate(all_groups_3d_merged):
        all_groups_3d_[m] = i
    parts = []
    for p in np.unique(all_groups_3d_):
        if p == -1:
            continue
        pid_idx = np.where(all_groups_3d_ == p)[0]
        pid_idx_remo = get_lcd(xyz[pid_idx])
        if pid_idx_remo is None:
            continue
        pid_idx = pid_idx[pid_idx_remo]
        parts.append(pid_idx)

    unlabel_seg = np.ones(xyz.shape[0], dtype=int) * -1
    cnt = 0
    for p in parts:
        unlabel_seg[p] = cnt
        cnt += 1
    return unlabel_seg


def predict(input_pc_file, category, part_names, model_dir="tmp"):
    print("[normalizing input point cloud...]")
    xyz, rgb = normalize_pc_o3d(input_pc_file)

    render_data_file = f"{model_dir}/mapping.npz"
    if os.path.exists(render_data_file):
        print("[existing mapping..]")
        pc_idx = np.load(render_data_file, allow_pickle=True)["pc_idx"]
        screen_coords = np.load(render_data_file, allow_pickle=True)["screen_coords"]
    else:
        print("[rendering point cloud...]")
        pc_idx, screen_coords = render_pc(xyz, rgb, device)
        np.savez_compressed(
            render_data_file, pc_idx=pc_idx, screen_coords=screen_coords
        )

    print("[predicting unlabeled seg...]")
    if unlabeled_seg:
        # pred load sam encoders' feat
        sam_encoders = {}
        for i in range(pc_idx.shape[0]):
            sam_encoders[i] = SamPredictor(sam)
            image = point_feature_mapping_view(pc_idx[i], feature=rgb)
            sam_encoders[i].set_image(image)
        groups_3d = {}
        starting_view_name_sequence = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]
        for starting_view_name in tqdm(
            starting_view_name_sequence, desc="getting part-level 2D groups"
        ):
            sam_predictor = sam_encoders[view_name.index(starting_view_name)]
            groups_3d_starting = generate_groups_3d_starting(
                starting_view_name,
                view_name,
                sam_predictor,
                xyz,
                rgb,
                pc_idx,
                screen_coords,
                device,
                model_dir,
            )
            groups_3d[starting_view_name] = groups_3d_starting
        
        # groups_3d (local) == extending == > groups_3d (global) (default: twice)
        print("extending...")
        for j in range(0, 2):
            for starting_view_name in groups_3d.keys():
                # getting the extension sequence
                extend_sequence = bfs_view_sequence(
                    view_adjacency_table, starting_view_name
                )[1:]
                extend_sequence_idx = [view_name.index(item) for item in extend_sequence]
                print(extend_sequence)
                for i, v_id in enumerate(extend_sequence_idx):
                    vi_groups_3d, mask_mean_score = single_view_extension(
                        groups_3d[starting_view_name],
                        pc_idx[v_id].squeeze(),
                        screen_coords[v_id].squeeze(),
                        xyz,
                        sam_encoders[v_id],
                        device,
                    )
                    groups_3d[starting_view_name] = vi_groups_3d

        # merging
        unlabel_seg = merging(groups_3d, T_merge, xyz)
        
        # save for unlabeled eval
        unlabeled_data_file = f"{model_dir}/unlabel_seg.npz"
        np.savez_compressed(unlabeled_data_file, unlabel_seg=unlabel_seg)

    # labeling for unlabeled part
    if with_labeling:
        print("labeling ...")
        unlabel_seg = np.load(f"{model_dir}/unlabel_seg.npz", allow_pickle=True)[
            "unlabel_seg"
        ]
        sem_seg, ins_seg = labeling(
            model_dir,
            xyz,
            rgb,
            pc_idx,
            screen_coords,
            unlabel_seg,
            part_names,
            num_views=pc_idx.shape[0],
            save_pred_img=False,
        )
        # save for ins and sem eval
        np.savez_compressed(f'{model_dir}/ins_sem_seg.npz',
                            sem_label=sem_seg, ins_label=ins_seg)

if __name__ == "__main__":
    
    meta = json.load(open("AKBSeg_meta.json"))
    data_root = "/mnt/sda/yuhengxue/reproduction_zerops_11_21/AKBSeg"
    categories = [
        "Drink",  # 51
        "Eyeglasses",  # 93
        "Faucet",  # 45
    ]
    all_models_cnt = sum([len(os.listdir(f"{data_root}/{c}")) for c in categories])
    count = 0
    start_time_all_akbseg = time.time()
    for category in categories:
        models = os.listdir(f"{data_root}/{category}")  # list of models
        models.sort()
        for model in models:
            # predict
            start_time = time.time()
            model_dir = f"{data_root}/{category}/{model}"
            print(f"当前模型:{model_dir}")
            predict(
                f"{model_dir}/pc.ply",
                category,
                meta[category],
                model_dir=model_dir,
            )
            # 记录程序的结束时间
            end_time = time.time()
            print(f"运行时间：{(end_time - start_time) / 60:.2f} min")
            count += 1
            print(f"已计算{count} / {all_models_cnt}")
    end_time_all_akbseg = time.time()

    # ============================================================================================ #
    # ============================================================================================ #

    meta = json.load(open("PartNetE_meta.json"))
    data_root = "/mnt/sda/yuhengxue/reproduction_zerops_11_21/PartNetE"
    categories = [
         "StorageFurniture",  # 338
        "TrashCan",  # 62
        "Box",  # 20
        "Bucket",  # 28
        "Camera",  # 29
        "Cart",  # 53
        "CoffeeMachine",  # 46
        "Dispenser",  # 49
    ]
    all_models_cnt = sum([len(os.listdir(f"{data_root}/{c}")) for c in categories])
    count = 0
    start_time_all_partnete = time.time()
    for category in categories:
        models = os.listdir(f"{data_root}/{category}")  # list of models
        models.sort()
        for model in models:
            # predict
            start_time = time.time()
            model_dir = f"{data_root}/{category}/{model}"
            print(f"当前模型:{model_dir}")
            predict(
                f"{model_dir}/pc.ply",
                category,
                meta[category],
                model_dir=model_dir,
            )
            # 记录程序的结束时间
            end_time = time.time()
            print(f"运行时间：{(end_time - start_time) / 60:.2f} min")
            count += 1
            print(f"已计算{count} / {all_models_cnt}")
    end_time_all_partnete = time.time()

    # ============================================================================================ #
    # ============================================================================================ #

    print(f"总运行时间：{(end_time_all_akbseg - start_time_all_akbseg) / 60:.2f} min")
    print(
        f"总运行时间：{(end_time_all_partnete - start_time_all_partnete) / 60:.2f} min"
    )

