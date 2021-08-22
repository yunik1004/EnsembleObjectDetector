import argparse
from dataset import CocoValData
from detector import EnsembleObjectDetector, ObjectDetector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble object detector")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device type")
    parser.add_argument(
        "--coco_config",
        default="configs/_base_/datasets/coco_detection.py",
        type=str,
        help="Path of the coco detection config",
    )
    parser.add_argument(
        "--model_config",
        default=[
            "configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
            "configs/retinanet/retinanet_r50_fpn_1x_coco.py",
            "configs/rpn/rpn_r50_fpn_1x_coco.py",
        ],
        type=str,
        nargs="+",
        help="List of the path of model configs",
    )
    parser.add_argument(
        "--model_checkpoint",
        default=[
            "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
            "checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
            "checkpoints/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth",
        ],
        type=str,
        nargs="+",
        help="List of the path of model checkpoints",
    )
    parser.add_argument(
        "--model_output",
        default=[
            "outputs/faster_rcnn_r50_fpn_1x_coco.pkl",
            "outputs/retinanet_r50_fpn_1x_coco.pkl",
            "outputs/rpn_r50_fpn_1x_coco.pkl",
        ],
        type=str,
        nargs="+",
        help="List of the path of model outputs",
    )
    args = parser.parse_args()

    num_model = len(args.model_config)
    assert (
        len(args.model_checkpoint) == num_model and len(args.model_output) == num_model
    )

    # Define COCO data
    data = CocoValData(args.coco_config)

    # Define detector models
    models = list()
    for i in range(num_model):
        detector = ObjectDetector(
            args.model_config[i],
            args.model_checkpoint[i],
            args.model_output[i],
            args.device,
        )
        models.append(detector)

    # Define ensemble model
    ensemble_detector = EnsembleObjectDetector(models)

    # Inference
    sub_results, ensemble_results = ensemble_detector.inference(data.data_loader)

    # Print evaluation metrics of submodels
    for i in range(num_model):
        mAP = data.evaluate(sub_results[i])
        print(f"\n>> mAP of Model {i} : {mAP * 100}\n")

    # Print evaluation metrics of ensemble model
    ensemble_mAP = data.evaluate(ensemble_results)
    print(f"\n>> mAP of Ensemble Model : {ensemble_mAP * 100}\n")
