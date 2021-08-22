from typing import Any, List
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset

# from mmdet.datasets.coco import CocoDataset


class CocoValData:
    """
    Class for treating the COCO style validation dataset
    """

    def __init__(self, config_path: str) -> None:
        """
        Constructor of CocoValData class.
        It generates the validation dataset and its dataloader.

        Parameters
        ----------
        config_path : str
            Path of the COCO style config
        """
        cfg = Config.fromfile(config_path)
        self._dataset = build_dataset(cfg.data.val)
        self._data_loader = build_dataloader(
            self._dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
        )

    @property
    def data_loader(self):
        """
        Getter of the attribute 'data_loader'

        Returns
        -------
        DataLoader
            Pytorch dataloader
        """
        return self._data_loader

    def evaluate(self, results: List[List[Any]]) -> float:
        """
        Evaluate the testing results.

        Parameters
        ----------
        results : List[List[torch.Tensor]]
            Testing results of the dataset

        Returns
        -------
        float
            Evaluated mAP
        """
        result = self._dataset.evaluate(results)
        return result["bbox_mAP"]
