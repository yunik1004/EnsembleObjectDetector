from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset

# from mmdet.datasets.coco import CocoDataset


def gen_val_data_loader(config_path: str):
    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    return data_loader
