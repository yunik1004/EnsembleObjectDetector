import os
from typing import Any, List
from mmdet.apis import init_detector, single_gpu_test
from mmcv.fileio import dump, load
from mmcv.parallel import MMDataParallel


class ObjectDetector:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        output_path: str,
        device: str = "cuda:0",
    ) -> None:
        model = init_detector(config_path, checkpoint_path, device=device)
        if device == "cpu":
            self._model = MMDataParallel(model)
        else:
            device_id = int(device[5:])
            self._model = MMDataParallel(model, device_ids=[device_id])

        self._output_path = output_path

    def inference(self, data_loader) -> List[Any]:
        if os.path.isfile(self._output_path):
            result = load(self._output_path)
        else:
            result = single_gpu_test(self._model, data_loader)
            dump(result, self._output_path)

        return result


class EnsembleObjectDetector:
    def __init__(self, detectors: List[ObjectDetector]) -> None:
        self._detectors = detectors

    def inference(self, data_loader) -> List[Any]:
        results = list()
        for detector in self._detectors:
            result = detector.inference(data_loader)
            results.append(result)

        return results
