import json
import os

data_roots = ["/mnt/sda/yuhengxue/reproduction_zerops_11_12/partnetE",
              "/mnt/sda/yuhengxue/reproduction_zerops_11_21/AKBSeg"]
metas = [json.load(open("PartNetE_meta.json", "r")), json.load(open("AKBSeg_meta.json", "r"))]

for idx, data_root in enumerate(data_roots):
    unique_file = []
    for category in metas[idx].keys():
        if not os.path.exists(f"{data_root}/{category}"):
            print(f"continue {category}")
            continue
        models = os.listdir(f"{data_root}/{category}")
        models.sort()
        for model in models:
            model_path = f"{data_root}/{category}/{model}"
            file_dir_names = os.listdir(model_path)
            for f in file_dir_names:
                if f not in unique_file:
                    unique_file.append(f)
    print(data_root)
    print(unique_file)  
