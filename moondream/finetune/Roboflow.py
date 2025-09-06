from roboflow import Roboflow
from torch.utils.data import Dataset
import json
from PIL import Image
import supervision as sv
import numpy as np
from collections import defaultdict
import random
import torch

def download_dataset(api_key, workspace, project_name, version_num):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    dataset = version.download("coco")
    return dataset

class RoboflowDataset(Dataset):
    def __init__(self, dataset_path, split):
        self.split = split

        sv_dataset = sv.DetectionDataset.from_coco(
            f"{dataset_path}/{split}/",
            f"{dataset_path}/{split}/_annotations.coco.json"
        )
        self.dataset = sv_dataset
        # self.dataset = self.dataset.shuffle(seed=111)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        filename, image, ann = self.dataset[idx] #ann.__class__ :supervision.detection.core.Detections        
        image = Image.fromarray(image)
        w,h = image.size
        target = ann.xyxy
        norm_target = []
        
        for x1,y1,x2,y2 in target:
            x1 = x1/w
            x2 = x2/w
            y1 = y1/h
            y2 = y2/h
            norm_target.append([x1,y1,x2,y2])
        
        class_id = ann.class_id
        bbox_area = ann.area
        
        class_id_to_bbox = {}
        for obj_class, bbox in zip(class_id, norm_target):
            class_id_to_bbox.setdefault(int(obj_class)-1, []).append(torch.tensor(bbox))

        # class_id = random.sample(list(class_id_to_bbox.keys()), 1)[0]
        return {
            "image": image,
            "class_to_bbox": class_id_to_bbox,
            "filename": filename,
            'size': (w,h),
        }