from roboflow import Roboflow
from torch.utils.data import Dataset, DataLoader
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


def custom_collate(batch):

    # image = [item['image'] for item in batch]
    # filename = [item['filename'] for item in batch]
    # class_to_bbox = [item['class_to_bbox'] for item in batch]
    
    data = {
        'image': batch[0]['image'],
        'filename': batch[0]['filename'],
        'size': batch[0]['size'],
        'class_to_bbox': batch[0]['class_to_bbox'],
    }
    return data



def get_loaders(datasets, BATCH_SIZE, NUM_WORKERS):
    
    train_loader = DataLoader(
        datasets['train'], 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        generator=torch.Generator(device='cuda'),
        num_workers=NUM_WORKERS,
        collate_fn = custom_collate,
        # pin_memory=True, # for faster data transfer to GPU
        # drop_last=True, # consistent batch sizes across GPUs
    )    

    val_loader = DataLoader(
        datasets['val'], 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn = custom_collate,
        num_workers=NUM_WORKERS, 
    )
    
    test_loader = DataLoader(
        datasets['test'], 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn = custom_collate,
        num_workers=NUM_WORKERS,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}