import argparse
from dataset import gen_data_loader
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
        "--model1_config",
        default="configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py",
        type=str,
        help="Path of the model 1 config",
    )
    parser.add_argument(
        "--model1_checkpoint",
        default="checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        type=str,
        help="Path of the model 1 checkpoint",
    )
    parser.add_argument(
        "--model1_output",
        default="outputs/faster_rcnn_r50_fpn_1x_coco.json",
        type=str,
        help="Path of the model 1 output",
    )
    parser.add_argument(
        "--model2_config",
        default="configs/retinanet/retinanet_r50_fpn_1x_coco.py",
        type=str,
        help="Path of the model 2 config",
    )
    parser.add_argument(
        "--model2_checkpoint",
        default="checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
        type=str,
        help="Path of the model 2 checkpoint",
    )
    parser.add_argument(
        "--model2_output",
        default="outputs/retinanet_r50_fpn_1x_coco.json",
        type=str,
        help="Path of the model 2 output",
    )
    parser.add_argument(
        "--model3_config",
        default="configs/rpn/rpn_r50_fpn_1x_coco.py",
        type=str,
        help="Path of the model 3 config",
    )
    parser.add_argument(
        "--model3_checkpoint",
        default="checkpoints/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth",
        type=str,
        help="Path of the model 3 checkpoint",
    )
    parser.add_argument(
        "--model3_output",
        default="outputs/rpn_r50_fpn_1x_coco.json",
        type=str,
        help="Path of the model 3 output",
    )
    args = parser.parse_args()

    # Define data loader
    coco_data_loader = gen_data_loader(args.coco_config)

    # Define detector models
    model1_detector = ObjectDetector(
        args.model1_config, args.model1_checkpoint, args.model1_output, args.device
    )

    model2_detector = ObjectDetector(
        args.model2_config, args.model2_checkpoint, args.model2_output, args.device
    )

    model3_detector = ObjectDetector(
        args.model3_config, args.model3_checkpoint, args.model3_output, args.device
    )

    # Define ensemble model
    ensemble_list = [model1_detector, model2_detector, model3_detector]
    ensemble_detector = EnsembleObjectDetector(ensemble_list)

    # Inference
    result = ensemble_detector.inference(coco_data_loader)
