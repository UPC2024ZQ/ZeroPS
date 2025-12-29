import os
import json
import open3d as o3d
import numpy as np
import shutil

data_root = "/mnt/sda/yuhengxue/reproduction_zerops_11_21/AKBSeg"

partnete_meta = json.load(open("AKBSeg_meta.json", "r"))

file_name_keyworlds = [
    'unlabel_seg',
    'ins_sem_seg'
]


for category in partnete_meta.keys():
    if not os.path.exists(f"{data_root}/{category}"):
        print(f"continue {category}")
        continue
    print(f"==============={category}====================")
    models = os.listdir(f"{data_root}/{category}")
    models.sort()

    for model in models:
        model_path = f"{data_root}/{category}/{model}"

        file_dir_names = os.listdir(model_path)
           
        for f_query in file_dir_names:
            for k in file_name_keyworlds:
                if k in f_query:
                    fill_path = f'{model_path}/{f_query}'
                    if os.path.isdir(fill_path):
                        shutil.rmtree(fill_path)  # Delete directory and all its contents
                    else:
                        os.remove(fill_path)  # Delete the file
                    print(fill_path)