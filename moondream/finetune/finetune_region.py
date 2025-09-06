import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import math
from safetensors.torch import save_file
import datasets

from tqdm import tqdm
from bitsandbytes.optim import AdamW
import wandb

from ..torch.weights import load_weights_into_model
from ..torch.moondream import MoondreamModel, MoondreamConfig, text_encoder
from ..torch.text import _produce_hidden
from ..torch.region import (
    decode_coordinate,
    decode_size,
    encode_coordinate,
    encode_size,
)

from .Roboflow import download_dataset, RoboflowDataset
from .mAP import per_object_mAP

# This is a intended to be a basic starting point. Your optimal hyperparams and data may be different.
MODEL_PATH = "/home/manugaur/moondream/models/model.safetensors"
LR = 1e-5
EPOCHS = 200
GRAD_ACCUM_STEPS = 10
STEP = 0
SAVE_METRIC = 0 #temp

def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2


def region_loss(
    hidden_states: torch.Tensor,
    w,
    labels: torch.Tensor,
    c_idx: torch.Tensor,
    s_idx: torch.Tensor,
):
    l_idx = torch.arange(len(labels))

    c_idx = c_idx - 1
    c_hidden = hidden_states[:, c_idx, :]
    c_logits = decode_coordinate(c_hidden, w)
    c_labels = labels[(l_idx % 4) < 2]

    c_loss = F.cross_entropy(
        c_logits.view(-1, c_logits.size(-1)),
        c_labels,
    )

    s_idx = s_idx - 1
    s_hidden = hidden_states[:, s_idx, :]
    s_logits = decode_size(s_hidden, w).view(-1, 1024)
    s_labels = labels[(l_idx % 4) >= 2]

    s_loss = F.cross_entropy(s_logits, s_labels)

    return c_loss + s_loss


class WasteDetection(Dataset):
    def __init__(self, split: str = "train"):
        self.dataset: datasets.Dataset = datasets.load_dataset(
            "moondream/waste_detection", split=split
        )
        self.dataset = self.dataset.shuffle(seed=111)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["image"]
        boxes = row["boxes"]
        labels = row["labels"]

        objects = {}
        for box, label in zip(boxes, labels):
            objects.setdefault(label, []).append(box)

        flat_boxes = []
        class_names = []
        for label, box_list in objects.items():
            for b in box_list:
                flat_boxes.append(b)
                class_names.append(label)

        flat_boxes = torch.as_tensor(flat_boxes, dtype=torch.float16)
        image_id = torch.tensor([idx], dtype=torch.int64)

        return {
            "image": image,
            "boxes": flat_boxes,
            "class_names": class_names,
            "image_id": image_id,
        }


import collections
def get_metric_summary(dataset_metric, name):
    global STEP
    class_stats = collections.defaultdict(lambda: {'sum': 0.0, 'count': 0})
    total_precision_sum = 0.0
    total_scores_count = 0

    for image_precisions in dataset_metric :
        for class_id, precision in image_precisions.items():
            class_stats[class_id]['sum'] += precision
            class_stats[class_id]['count'] += 1
            total_precision_sum += precision
            total_scores_count += 1

    avg_dataset_precision = total_precision_sum / total_scores_count if total_scores_count > 0 else 0
    # print(f"AVG {name}: {avg_dataset_precision:.4f}")
    avg_class_precision = {}
    for class_id in sorted(class_stats.keys()):
        stats = class_stats[class_id]
        average = stats['sum'] / stats['count'] if stats['count'] > 0 else 0
        avg_class_precision[f"{name}/{class_id}"] = average
        # print(f"{class_id}: {average:.4f} ({stats['count']} counts)")
    # print("\n")
    avg_class_precision[f"{name}/AVG"] = avg_dataset_precision
    wandb.log(avg_class_precision, step=STEP)
    return avg_dataset_precision
    
@torch.no_grad()
def eval_obj_det(model, val_dataset):
    global STEP
    global SAVE_METRIC
    model.eval()
    AP = [] #image_idx --> AP
    AP_50 = []
    AR_100 = []
    
    for sample in val_dataset:
        AP_dict = {}
        AP_50_dict= {}
        AR_100_dict= {}
        for class_name, boxes_list in sample['class_to_bbox'].items():
            instruction = f"\n\nDetect: {class_name}\n\n"
            out = model.detect(sample['image'], instruction)['objects']
            ######## TO DO --> plot 5-10 outputs
            metadata = {
                "size" : sample['size'],
                "prompt": instruction,
                'filename': sample['filename'],
            }
            metrics = per_object_mAP([_.tolist() for _ in boxes_list], [list(_.values()) for _ in out], metadata)
            AP_50_dict.update({class_name:float(metrics['AP_50'])})
            AR_100_dict.update({class_name:float(metrics['AR_100'])})
            AP_dict.update({class_name:float(metrics['AP'])})
        
        AP.append(AP_avg_dict)
        AP_50.append(AP_50_dict)
        AR_100.append(AR_100_dict)

    _ = get_metric_summary(AP, 'AP (0.50:0.95)')
    avg_AP_50 = get_metric_summary(AP_50, 'AP (0.50)')
    avg_AR_100 = get_metric_summary(AR_100, 'AR (0.50:0.95) | 100 Det')
    
    # if avg_AP_50 > SAVE_METRIC:
    #     SAVE_METRIC = avg_AP_50
    #     save_file(
    #     model.state_dict(),
    #     "checkpoints/moondream_finetune.safetensors",
    #     )

    model.train()

def main():
    global STEP
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    wandb.init(
        project="moondream-ft",
        config={
            "EPOCHS": EPOCHS,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
        },
    )

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(MODEL_PATH, model)

    # If you are struggling with GPU memory, try AdamW8Bit
    optimizer = AdamW(
        [{"params": model.region.parameters()}],
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    # dataset = WasteDetection()
    
    ds = download_dataset("4BDHggHM6vkVOoK3g0s3", "objectdetvlm", "water-meter-jbktv-7vz5k-fsod-ftoz-qii9s", 1)
    datasets = {
        "train": RoboflowDataset(ds.location,"train"),
        "val": RoboflowDataset(ds.location,"valid"),
        "test": RoboflowDataset(ds.location,"test"),
    }
    dataset = datasets['train']
    
    # init evals
    eval_obj_det(model, datasets['val'])
    
    total_steps = EPOCHS * len(dataset) // GRAD_ACCUM_STEPS
    pbar = tqdm(total=total_steps)
    i=0
    #### Training
    for epoch in range(EPOCHS):
        for sample in dataset:
            i+=1
            STEP+=1
            with torch.no_grad():
                img_emb = model._run_vision_encoder(sample["image"])
                bos_emb = text_encoder(
                    torch.tensor(
                        [[model.config.tokenizer.bos_id]], device=model.device
                    ),
                    model.text,
                )
                eos_emb = text_encoder(
                    torch.tensor(
                        [[model.config.tokenizer.eos_id]], device=model.device
                    ),
                    model.text,
                )
            
            # boxes_by_class = {}
            # for box, cls in zip(sample["boxes"], sample["class_names"]):
            #     boxes_by_class.setdefault(cls, []).append(box)
            
            total_loss = 0.0
            boxes_by_class = sample['class_to_bbox']
            for class_name, boxes_list in boxes_by_class.items():
            # for class_name, boxes_list in sample['class_to_bbox'].items():
                with torch.no_grad():
                    instruction = f"\n\nDetect: {class_name}\n\n"
                    instruction_tokens = model.tokenizer.encode(instruction).ids
                    instruction_emb = text_encoder(
                        torch.tensor([[instruction_tokens]], device=model.device),
                        model.text,
                    ).squeeze(0)

                cs_emb = []
                cs_labels = []
                c_idx = []
                s_idx = []
                for bb in boxes_list:
                    l_cs = len(cs_emb)
                    cs_emb.extend(
                        [
                        encode_coordinate(bb[0].to(model.region.coord_features.dtype).unsqueeze(0), model.region),
                        encode_coordinate(bb[1].to(model.region.coord_features.dtype).unsqueeze(0), model.region),
                        encode_size(bb[2:4].to(model.region.coord_features.dtype).unsqueeze(0), model.region).squeeze(0),
                        ]
                    )
                    c_idx.extend([l_cs, l_cs + 1])
                    s_idx.append(l_cs + 2)

                    # Create coordinate bin labels
                    coord_labels = [
                        min(max(torch.round(p * 1023), 0), 1023).item() for p in bb[:2]
                    ]

                    # Create size bin labels using log-scale mapping
                    s_log2_bins = []
                    for s_val in bb[2:4]:
                        s_val = float(s_val)
                        s_clamped = max(s_val, 1 / 1024)
                        s_log2 = math.log2(s_clamped)
                        mapped = (s_log2 + 10.0) / 10.0 * 1023.0
                        s_bin = int(round(mapped))
                        s_bin = max(min(s_bin, 1023), 0)
                        s_log2_bins.append(s_bin)

                    # Combine coordinate and size bin labels
                    cs_labels.extend(coord_labels + s_log2_bins)

                if len(cs_emb) == 0:
                    continue
                cs_emb = torch.stack(cs_emb)

                inputs_embeds = torch.cat(
                    [bos_emb, img_emb[None], instruction_emb, cs_emb[None], eos_emb],
                    dim=1,
                )
                prefix = inputs_embeds.size(1) - cs_emb.size(0)
                c_idx = torch.tensor(c_idx) + prefix
                s_idx = torch.tensor(s_idx) + prefix

                hidden = _produce_hidden(
                    inputs_embeds=inputs_embeds, w=model.text, config=config.text
                )

                loss = region_loss(
                    hidden_states=hidden,
                    w=model.region,
                    labels=torch.tensor(cs_labels, dtype=torch.int64),
                    c_idx=c_idx,
                    s_idx=s_idx,
                )
                total_loss += loss

            total_loss.backward()

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr_val = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_val
                pbar.set_postfix(
                    {"step": i // GRAD_ACCUM_STEPS, "loss": total_loss.item()}
                )
                pbar.update(1)
                wandb.log(
                    {
                        "loss/train": total_loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }, step=STEP
                )
    
        if (epoch+1) % 10 == 0:
            eval_obj_det(model, datasets['val'])
    
    wandb.finish()

if __name__ == "__main__":
    """
    Replace paths with your appropriate paths.
    To run: python -m moondream.finetune.finetune_region
    """
    main()
