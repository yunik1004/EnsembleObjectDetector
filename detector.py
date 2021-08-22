import os
from typing import Any, List, Tuple
from mmdet.apis import init_detector, single_gpu_test
from mmcv.fileio import dump, load
from mmcv.parallel import MMDataParallel


class ObjectDetector:
    """
    Object detector model
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        output_path: str,
        device: str = "cuda:0",
    ) -> None:
        """
        Constructor of ObjectDetector class.
        It generates the pretrained Pytorch model based on the config and checkpoint.

        Parameters
        ----------
        config_path : str
            Path of the config
        checkpoint_path : str
            Path of the checkpoint
        output_path : str
            Path of the output
        device : str, optional
            Device type. It should be "cpu" or "cuda:xxx", by default "cuda:0"
        """
        model = init_detector(config_path, checkpoint_path, device=device)
        if device == "cpu":
            self._model = MMDataParallel(model)
        else:
            device_id = int(device[5:])
            self._model = MMDataParallel(model, device_ids=[device_id])

        self._output_path = output_path

    def inference(self, data_loader) -> List[List[Any]]:
        """
        Returns the list of inference results for the given data loader.
        It also saves the results into the output file.

        Parameters
        ----------
        data_loader : DataLoader
            Pytorch dataloader

        Returns
        -------
        List[List[torch.Tensor]]
            Inference results
        """
        if os.path.isfile(self._output_path):
            result = load(self._output_path)
        else:
            result = single_gpu_test(self._model, data_loader)
            dump(result, self._output_path)

        return result


class EnsembleObjectDetector:
    """
    Object detector model which uses the ensemble method
    """

    def __init__(self, detectors: List[ObjectDetector]) -> None:
        """
        Constructor of EnsembleObjectDetector class.
        It stores ObjectDetector objects which will be used for the ensemble model.

        Parameters
        ----------
        detectors : List[ObjectDetector]
            List of ObjectDetector objects which are the part of ensemble model
        """
        self._detectors = detectors

    def inference(self, data_loader) -> Tuple[List[List[List[Any]]], List[List[Any]]]:
        """
        Return the inference results for the given data loader.

        Parameters
        ----------
        data_loader : DataLoader
            Pytorch dataloader

        Returns
        -------
        Tuple[List[List[List[torch.Tensor]]], List[List[torch.Tensor]]
            First element: List of submodel results
            Second element: Ensemble inference results
        """
        sub_results = list()
        for detector in self._detectors:
            result = detector.inference(data_loader)
            sub_results.append(result)

        # TODO : Generate ensemble results
        ensemble_results = sub_results[0]

        return sub_results, ensemble_results
