## Installation

### Create a conda envrionment

```
conda env create -f environment.yml
conda activate zerops
```

### Install GLIP
We use PartSLIP's [*modified version*](https://github.com/Colin97/GLIP).
```
cd GLIP
python setup.py build develop --user
```

### Install PyTorch3D
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Install Point2_ops_lib
```
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

### Download pretrained checkpoints   
Please download checkpoints to `models/`.   
For SAM, please use the pre-trained checkpoint [`sam_vit_h_4b8939.pth`](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).   
For GLIP, please use the pre-trained checkpoint [`glip_large_model.pth`](https://huggingface.co/datasets/minghua/PartSLIP/tree/main/models).

## Quick-Demo

```
python zerops_demo.py --object_name Chair --part_names arm back leg seat wheel --with_knn --with_labeling
```

## Run on PartNetE and AKBSeg

To efficiently utilize GPU memory on a 24GB GPU, you can run the following commands across three separate tmux sessions. 

**tmux 1**:
```
   python zerops_split1.py --unlabeled_seg && python zerops_split1.py --with_labeling
```
**tmux 2**:
```
   python zerops_split2.py --unlabeled_seg && python zerops_split2.py --with_labeling
```
**tmux 3**:
```
   python zerops_split3.py --unlabeled_seg && python zerops_split3.py --with_labeling
```
## Evaluation
```
   python -u eval_unseg.py
```
```
   python -u eval_3d_ins.py
```
```
   python -u eval_3d_sem.py
```

## Tips:
1. We follow PartSLIP's work to foucs **dense and colored** point cloud. If your dense point cloud does not have color information, you can use the normals as RGB values. 
2. To achieve 2D-3D mapping with PyTorch3D rendering, it is important to note that the results are not always **100% accurate**. Therefore, we apply a connected component preservation operation in both the 2D and 3D spaces. In the 3D space, we use DBSCAN, which has two key parameters: min_points and eps. The default values of these parameters are currently set to work with PartNetE and AKBSeg. However, if you intend to apply this method to other data, it may be necessary to adjust these parameters accordingly.
3. The core of this work is to validate the effectiveness of the extension. The current viewpoint settings (both in terms of number and placement) may not be optimal. We believe that exploring how to arrange the viewpoints to achieve higher segmentation performance and efficiency for Zerops is a valuable direction for further investigation.
4. The merging algorithm may represent the optimal approach for performing zero-shot 3D part segmentation using SAM, as it essentially lift 2D coverage (from fine to coarse granularity) into 3D. Therefore, we believe that proving Zerops as the best choice for zero-shot 3D part segmentation using SAM is an intriguing idea.

## You can contact us via the following email addresses:

yuhengxue@nuist.edu.cn

yuhengxue321@gmail.com

luis0907@qq.com



