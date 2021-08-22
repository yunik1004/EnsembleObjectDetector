import os
from typing import List
from ensemble_boxes import weighted_boxes_fusion
from mmcv.fileio import dump, load
from mmcv.parallel import MMDataParallel
from mmdet.apis import init_detector, single_gpu_test
import numpy as np


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

    def inference(self, data_loader):
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

    def inference(self, data_loader):
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

        # Generate ensemble results using WBF
        ensemble_results = list()

        num_model = len(self._detectors)
        num_class = len(sub_results[0][0])

        for d, data in enumerate(data_loader):
            img_shape = data["img_metas"][0].data[0][0]["ori_shape"]
            img_x = img_shape[1]
            img_y = img_shape[0]

            img_scaler = np.array([img_x, img_y, img_x, img_y])

            # Convert the result into WBF's format
            box_list = list()
            score_list = list()
            label_list = list()

            for m in range(num_model):
                boxes = list()
                scores = list()
                classes = list()

                for c in range(num_class):
                    tmp = sub_results[m][d][c]
                    tmp_box = np.clip(tmp[:, :4] / img_scaler, 0, 1)
                    tmp_score = tmp[:, 4]
                    tmp_class = np.full_like(tmp_score, c, dtype=int)

                    boxes.append(tmp_box)
                    scores.append(tmp_score)
                    classes.append(tmp_class)

                box_list.append(np.concatenate(boxes, axis=0))
                score_list.append(np.concatenate(scores, axis=0))
                label_list.append(np.concatenate(classes, axis=0))

            # Run WBF
            ensemble_boxes, ensemble_scores, ensemble_labels = weighted_boxes_fusion(
                box_list, score_list, label_list
            )

            # Restore the result format
            ensemble_tmp = np.column_stack(
                [ensemble_boxes * img_scaler, ensemble_scores]
            )
            ensemble_labels = ensemble_labels.astype(int)

            ensemble_result = [list() for _ in range(num_class)]
            for i, c in enumerate(ensemble_labels):
                ensemble_result[c].append(ensemble_tmp[i])
            for c in range(num_class):
                ensemble_result[c] = np.array(ensemble_result[c])

            ensemble_results.append(ensemble_result)

        return sub_results, ensemble_results
