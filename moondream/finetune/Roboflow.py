from roboflow import Roboflow
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import supervision as sv
import numpy as np
from collections import defaultdict
import random
import torch
import torchvision.transforms as T

def train_transform():
    return T.Compose([
        T.RandomApply([T.ColorJitter(brightness=.5, hue=.4)], p=0.7),
        T.RandomGrayscale(p=0.4),
    ])

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
        self.transform = train_transform() if split =="train" else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        filename, image, ann = self.dataset[idx] #ann.__class__ :supervision.detection.core.Detections        
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
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


################   Augmentations   ################

def eval_transform(img_res):
    return T.Compose([
        T.Resize(size=(img_res, img_res), interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
        T.Lambda(img2rgb),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    
def GeometricTransform(image, heatmap, img_res, pflip=0.2, protate=0):
    og_heatmap_size = heatmap.size(-1)
    
    # Resize
    resize = T.Resize(size=(img_res, img_res), interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True)
    image = resize(image)
    # heatmap = resize(heatmap)

    # # Random crops
    # i, j, h, w = T.RandomCrop.get_params(
    #     image, output_size=(512, 512))
    # image = TF.crop(image, i, j, h, w)
    # heatmap = TF.crop(heatmap, i, j, h, w)

    ## Random horizontal flipping
    if random.random() < pflip:
        image = TF.hflip(image)
        heatmap = TF.hflip(heatmap)

    ## Random vertical flipping
    if random.random() < pflip:
        image = TF.vflip(image)
        heatmap = TF.vflip(heatmap)

    ## Random rotation
    # if random.random() < protate:
    #     angle = random.randint(-180, 179)
    #     image = TF.rotate(image, angle)
    #     heatmap = TF.rotate(heatmap, angle)

    return image, heatmap.flatten() #, T.Resize(size=(og_heatmap_size, og_heatmap_size), interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True)(heatmap).squeeze(0).flatten()

def train_transform():
    return T.Compose([
        T.RandomApply([T.ColorJitter(brightness=.5, hue=.3)], p=0.7),
        T.RandomGrayscale(p=0.1),
        # T.Lambda(img2rgb),
        # T.ToTensor(),
        # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
