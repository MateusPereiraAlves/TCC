from models.component_segmentaion import grounding_segmentation
import os
import glob
import yaml


mask_path = "./masks/mvtec"
categories = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


for category in categories:
    image_paths = sorted(glob.glob(f"./data/mvtec/{category}/test/*/*.png"))
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

for category in categories:
    image_paths = sorted(glob.glob(f"./data/mvtec/{category}/train/*/*.png"))
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

mask_path = "./masks/VisA_pytorch/1cls"
categories = [
    "candle",
    "capsules",
    "chewinggum",
    "cashew",
    "fryum",
    "pipe_fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
]


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


for category in categories:
    image_paths = sorted(glob.glob(f"./data/VisA_pytorch/1cls/{category}/test/*/*.JPG"))
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

for category in categories:
    image_paths = sorted(glob.glob(f"./data/VisA_pytorch/1cls/{category}/train/*/*.JPG"))
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

mask_path = "./masks/mvtec_loco_caption"
categories = [
    'breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors'
]


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


for category in categories:
    image_paths = sorted(glob.glob(f"./data/mvtec_loco_caption/{category}/test/*/*.png"))
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

for category in categories:
    image_paths = sorted(glob.glob(f"./data/mvtec_loco_caption/{category}/train/*/*.png"))
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

mask_path = "./masks"
categories = [
    "LiverCT",
    "BrainMRI",
    "RESC",
    "HIS",
    "ChestXray"
]


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


for category in categories:
    image_paths = sorted(glob.glob(f"./data/{category}/test/*/*.png"))
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

for category in categories:
    image_paths = sorted(glob.glob(f"./data/{category}/train/*/*.png"))
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

mask_path = "./masks"
categories = [
    "OCT17"
]


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


for category in categories:
    image_paths = sorted(glob.glob(f"./data/{category}/test/*/*.jpeg"))
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

for category in categories:
    image_paths = sorted(glob.glob(f"./data/{category}/train/*/*.jpeg"))
    config = read_config(f"./configs/class_histogram/{category}.yaml")
    os.makedirs(f"{mask_path}/{category}", exist_ok=True)
    grounding_segmentation(
        image_paths, f"{mask_path}/{category}", config["grounding_config"]
    )

